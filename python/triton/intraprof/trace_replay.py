import copy
import logging
from dataclasses import dataclass, field
from inspect import signature
from os import name
from typing import Dict, List, Optional

import torch
import triton


word_per_slot = 2
warp_per_group = 4
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trace_replay")


chrome_colors = [
    "cq_build_passed",
    "cq_build_failed",
    "thread_state_iowait",
    "thread_state_running",
    "thread_state_runnable",
    "thread_state_unknown",
    "rail_response",
    "rail_idle",
    "rail_load",
    "cq_build_attempt_passed",
    "cq_build_attempt_failed",
]


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


@dataclass
class IntraKernelConfig(object):
    num_warps: int
    proton_slots: int
    granularity: str = ""
    names: Optional[Dict[int, str]] = field(default=None)
    kernel_file: Optional[str] = field(default=None)


@dataclass
class ProfileEvent(object):
    region_id: int
    start: int
    end: int
    wg: int
    signature: int
    hwid: int
    block_id: int
    syn: bool
    color: str


def _get_signature_from_metadata(metadata):
    return (metadata & 0x7FFFFFFF) >> 16


def _is_signature_from_hwid(sig, hwid):
    hwid_sig = hwid & 0xFFF
    return sig == hwid_sig


def _get_popular_signature(data, size):
    signatures = []
    for i in range(0, size, word_per_slot):
        metadata = data[i]
        signatures.append(_get_signature_from_metadata(metadata))
    return max(signatures, key=signatures.count)


def _print(events: List[ProfileEvent]):
    for event in events:
        print(
            f"region_id={event.region_id}, start={event.start}, end={event.end}, wg={event.wg}"
        )


def _parse_record(
    metadata: int,
    cycle: int,
    event: ProfileEvent,
    parse_state: Dict[int, str],
    wg_signature: Optional[int],
    event_list: List[ProfileEvent],
    msg_suffix: str,
):
    is_start = False if metadata >> 31 == 1 else True
    region_id = metadata & 0x0000FFFF
    assert region_id < 65536
    signature = _get_signature_from_metadata(metadata)
    prev_state = parse_state.get(region_id, "init")

    # We check clock overflow (32-bit) and ignore the extra event.
    # For example, two consecutive start events of the same region in the event stream.
    if is_start:
        if prev_state in ["init", "end"]:
            event.start = cycle
            parse_state[region_id] = "start"
        else:
            logger.debug("Ignore an extra START record" + msg_suffix)
    else:
        if prev_state == "start":
            event.end = cycle
            parse_state[region_id] = "end"
            if event.end < event.start:
                logger.debug("Ignore an event due to clock overflow" + msg_suffix)
                return
            # Check signature consistency (for AMD GPUs, we rely on the round-robin scheduling of the wavefronts)
            if (wg_signature is not None) and (wg_signature != signature):
                logger.warning(
                    "Synthesize a record, might not be accurate " + msg_suffix
                )
                event.syn = True
            event_list.append(copy.deepcopy(event))
        else:
            logger.debug("Ignore an extra END record" + msg_suffix)


def _get_wg_events(warp_meta: List[int], wg_id: int, data: List[int], block_id: int):
    assert word_per_slot == 2
    index = warp_meta[1]
    size = index if len(data) > index else len(data)
    event_list = []
    active_event = {}
    # Each region has 3 parsing states: init, start, end
    parse_state = {}
    wg_signature = None
    hwid = warp_meta[0]
    color_idx = 0
    colors = {}

    if is_hip():
        wg_signature = _get_popular_signature(data, size)
        for i in range(warp_per_group):
            hwid = (
                warp_meta[2 * i]
                if _is_signature_from_hwid(wg_signature, warp_meta[2 * i])
                else hwid
            )

    for i in range(0, size, word_per_slot):
        metadata = data[i]
        cycle = data[i + 1]
        region_id = metadata & 0x0000FFFF

        # Assign colors to regions
        if region_id not in colors:
            colors[region_id] = chrome_colors[color_idx]
            color_idx = (color_idx + 1) % len(chrome_colors)

        signature = _get_signature_from_metadata(metadata)

        if region_id not in active_event:
            active_event[region_id] = ProfileEvent(
                region_id,
                0,
                0,
                wg_id,
                signature,
                hwid,
                block_id,
                False,
                colors[region_id],
            )

        event = active_event[region_id]

        suffix = f"(region={region_id}, wg={wg_id}, block={block_id})"

        _parse_record(
            metadata, cycle, event, parse_state, wg_signature, event_list, suffix
        )

    return event_list


def _get_warp_events(warp_meta: List[int], wg_id: int, data: List[int], block_id: int):
    assert word_per_slot == 2
    index = warp_meta[1]
    size = index if len(data) > index else len(data)
    event_list = []
    active_event = [{} for i in range(warp_per_group)]
    # Each region has 3 parsing states: init, start, end
    parse_state = [{} for i in range(warp_per_group)]
    slot_per_warpgroup = warp_per_group * word_per_slot
    warp_signature = None
    colors = {}
    color_idx = 0

    for i in range(0, size, slot_per_warpgroup):
        for j in range(0, slot_per_warpgroup, word_per_slot):
            wid_in_wg = j // word_per_slot
            hwid = warp_meta[2 * wid_in_wg]
            metadata = data[i + j]
            cycle = data[i + j + 1]
            region_id = metadata & 0x0000FFFF

            # Assign colors to regions
            if region_id not in colors:
                colors[region_id] = chrome_colors[color_idx]
                color_idx = (color_idx + 1) % len(chrome_colors)

            signature = _get_signature_from_metadata(metadata)
            warp_id = wg_id * warp_per_group + wid_in_wg

            if region_id not in active_event[wid_in_wg]:
                active_event[wid_in_wg][region_id] = ProfileEvent(
                    region_id,
                    0,
                    0,
                    warp_id,
                    signature,
                    hwid,
                    block_id,
                    False,
                    colors[region_id],
                )

            event = active_event[wid_in_wg][region_id]
            suffix = f"(region={region_id}, warp={warp_id}, block={block_id})"

            _parse_record(
                metadata,
                cycle,
                event,
                parse_state[wid_in_wg],
                warp_signature,
                event_list,
                suffix,
            )

    return event_list


def _shift_start(event_list: List[ProfileEvent]):
    start_time = [e.start for e in event_list]

    if len(start_time) == 0:
        return

    min_start = min(start_time)
    for event in event_list:
        event.start -= min_start
        event.end -= min_start


def _single_block_memsize(config: IntraKernelConfig):
    # preample(1), blockid (1), [hwid, index] (2 * num_warps)
    header_size = 2 + config.num_warps * 2
    return header_size + config.proton_slots * word_per_slot


def intra_kernel_memsize(num_blocks: int, config: IntraKernelConfig):
    return num_blocks * _single_block_memsize(config)


def _get_chrome_event_str(event: ProfileEvent, config: IntraKernelConfig):
    if config.names:
        name = config.names.get(event.region_id, f"region_{event.region_id}")
    else:
        name = f"region_{event.region_id}"

    if is_cuda():
        return f'{{"cname": "{event.color}", "name": "{name}", "cat": "triton", \
            "ph": "X", "ts": {event.start}, "dur": {event.end - event.start}, \
            "pid": "threadblock {event.block_id}", "tid": "{config.granularity} {event.wg}", \
            "args":{{"sm_id": "{event.hwid}", "Note": "0.001ms = 1 cycle"}}}}'
    elif is_hip():
        if event.syn:
            hwid = "SYNTHESIZED"
        else:
            hwid = event.hwid
        hwreg = event.hwid
        wave_id = hwreg & 0xF
        simd_id = (hwreg >> 4) & 0x3
        pipe_id = (hwreg >> 6) & 0x3
        wavefront_id = (
            event.wg * warp_per_group if config.granularity == "warpgroup" else event.wg
        )
        return f'{{"cname": "{event.color}", "name": "{name}", "cat": "triton", \
            "ph": "X", "ts": {event.start}, "dur": {event.end - event.start}, \
            "pid": "workgroup {event.block_id}", "tid": "wavefront {wavefront_id} [SIMD{simd_id}, SLOT{wave_id}]", \
            "args":{{"hw_id": "{hwid}", "wave_id": "{wave_id}", "simd_id":"{simd_id}", "pipe_id":"{pipe_id}", "Note": "0.001ms = 1 cycle"}}}}'
    else:
        raise ValueError("Invalid backend")


def _get_const_overhead(config):
    if is_hip():
        if config.granularity == "warpgroup":
            return 56
        return 36
    elif is_cuda():
        return 15
    return 0


def _shift_cost(events: List[ProfileEvent], cost: int, time_shift: int, warp: int):
    for e in events:
        if e.wg != warp:
            continue

        if e.start >= time_shift:
            e.start -= cost
        if e.end >= time_shift:
            e.end -= cost


def _reduce_const_overhead(events: List[ProfileEvent], config: IntraKernelConfig):
    size = len(events)
    cost = _get_const_overhead(config)

    for i in range(size):
        time_shift = events[i].start
        warp = events[i].wg
        _shift_cost(events, cost, time_shift, warp)

        time_shift = events[i].end
        _shift_cost(events, cost, time_shift, warp)

        if events[i].end - events[i].start < cost:
            logger.debug("Adjust event time too small: " + str(events[i]))
            events[i].end = events[i].start + cost - 2


def get_event_list(
    block_num: int, config: IntraKernelConfig, profile_mem: torch.Tensor
):
    assert config.granularity in ["warpgroup", "warp"]

    scratch = _single_block_memsize(config)
    warp_meta_size = 2 * config.num_warps
    event_list = []
    for i in range(block_num):
        block_event_list = []
        workspace = profile_mem[i * scratch : (i + 1) * scratch]
        preample = workspace[0].item()
        block_id = workspace[1].item()

        if preample != 0xDEADBEEF:
            logger.debug("Invalid preample in block " + str(block_id) + ", skip it")
            continue

        warp_meta = workspace[2 : 2 + warp_meta_size].tolist()
        data = workspace[2 + warp_meta_size :].tolist()
        wg_num = config.num_warps // warp_per_group
        assert len(data) == config.proton_slots * word_per_slot
        wg_data_len = len(data) // wg_num

        for j in range(wg_num):
            ws = j * wg_data_len
            warp_meta_wg = warp_meta[
                j * warp_per_group * 2 : j * warp_per_group * 2 + 2 * warp_per_group
            ]
            if config.granularity == "warpgroup":
                wg_events = _get_wg_events(
                    warp_meta_wg, j, data[ws : ws + wg_data_len], block_id
                )
            elif config.granularity == "warp":
                wg_events = _get_warp_events(
                    warp_meta_wg, j, data[ws : ws + wg_data_len], block_id
                )
            else:
                raise ValueError("Invalid granularity: " + config.granularity)

            _reduce_const_overhead(wg_events, config)
            block_event_list += wg_events

        _shift_start(block_event_list)

        event_list += block_event_list

    return event_list


def dump_chrome_trace(
    block_num: int, config: IntraKernelConfig, profile_mem: torch.Tensor, file_name: str, compiled_fn
):
    metadata = compiled_fn.metadata
    if metadata.proton_granularity == 1:
        config.granularity = "warpgroup"
    else:
        config.granularity = "warp"

    if (metadata.num_warps != config.num_warps):
        raise ValueError("num_warps in IntraKernelConfig should be: " + str(metadata.num_warps))

    if block_num > 256:
        print("Warning: block_num > 256, we only print the first 256 blocks")
        block_num = 256

    event_list = get_event_list(block_num, config, profile_mem)
    trace_str = '{"traceEvents": ['
    for event in event_list:
        chrome_event_str = _get_chrome_event_str(event, config)
        trace_str += chrome_event_str + ",\n"
    trace_str = trace_str[:-2] + "]}"

    with open(file_name, "w") as f:
        f.write(trace_str)


def const_grid(grid, autotune_configs, func_args, **kwargs):
    if callable(grid):
        if len(autotune_configs) != 1:
            raise ValueError("Only one autotune config is supported")

        args = autotune_configs[0].all_kwargs()
        args.update(func_args)
        args.update(kwargs)
        const_grid = grid(args)
    else:
        const_grid = grid

    return const_grid