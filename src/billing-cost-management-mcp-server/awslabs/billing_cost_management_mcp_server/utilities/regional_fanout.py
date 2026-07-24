# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for bounded fan-out across AWS regions."""

import asyncio
import base64
import binascii
import json
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    cast,
)


RequestState = TypeVar('RequestState')
Success = TypeVar('Success')
Error = TypeVar('Error')


@dataclass
class RegionalFanoutResult(Generic[Success, Error]):
    """Outcomes from querying a set of regions."""

    successes: Dict[str, Success]
    errors: Dict[str, Error]
    misses: List[str]


class RegionalTokenError(ValueError):
    """A structured validation failure for an opaque regional pagination token."""

    def __init__(self, reason: str, **details: Any):
        """Initialize the error with a stable reason and contextual details."""
        super().__init__(reason)
        self.reason = reason
        self.details = details


async def fan_out_regions(
    requests: Mapping[str, RequestState],
    worker: Callable[[str, RequestState], Awaitable[Success]],
    format_error: Callable[[str, Exception], Awaitable[Error]],
    *,
    max_concurrency: int,
    is_miss: Optional[Callable[[Exception], bool]] = None,
) -> RegionalFanoutResult[Success, Error]:
    """Execute one worker per region with bounded concurrency.

    The utility owns execution mechanics only. Callers decide how to create clients,
    classify misses, format errors, and merge successful values.

    Args:
        requests: Region keys mapped to caller-defined request state.
        worker: Async callable that executes one regional request.
        format_error: Async callable that turns an exception into a caller-defined error.
        max_concurrency: Maximum number of regional workers running concurrently.
        is_miss: Optional exception classifier for expected regional misses.

    Returns:
        Regional successes, errors, and misses, each preserving request order.

    Raises:
        ValueError: If max_concurrency is less than one.
    """
    if max_concurrency < 1:
        raise ValueError('max_concurrency must be at least 1')

    semaphore = asyncio.Semaphore(max_concurrency)

    async def query_region(
        region: str, request_state: RequestState
    ) -> Tuple[str, Optional[Success], Optional[Error], bool]:
        async with semaphore:
            try:
                value = await worker(region, request_state)
                return region, value, None, False
            except Exception as error:
                if is_miss is not None and is_miss(error):
                    return region, None, None, True
                formatted_error = await format_error(region, error)
                return region, None, formatted_error, False

    outcomes = await asyncio.gather(
        *(query_region(region, state) for region, state in requests.items())
    )

    successes: Dict[str, Success] = {}
    errors: Dict[str, Error] = {}
    misses: List[str] = []
    for region, value, error, missed in outcomes:
        if missed:
            misses.append(region)
        elif error is not None:
            errors[region] = error
        else:
            successes[region] = cast(Success, value)

    return RegionalFanoutResult(successes=successes, errors=errors, misses=misses)


def encode_regional_next_token(region_next_tokens: Mapping[str, str]) -> str:
    """Encode per-region pagination state as one opaque token."""
    payload = json.dumps(region_next_tokens, separators=(',', ':'), sort_keys=True)
    return base64.b64encode(payload.encode('utf-8')).decode('ascii')


def decode_regional_next_token(
    next_token: Optional[str], supported_regions: List[str]
) -> Dict[str, Optional[str]]:
    """Decode and validate opaque per-region pagination state.

    Omitting the token starts a request in every supported region.

    Raises:
        RegionalTokenError: If the token is malformed or contains invalid state.
    """
    if not next_token:
        return dict.fromkeys(supported_regions)

    try:
        decoded = base64.b64decode(next_token.encode('ascii'), validate=True).decode('utf-8')
        parsed = json.loads(decoded)
    except (UnicodeEncodeError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError) as error:
        raise RegionalTokenError('decode_error', cause=str(error)) from error

    if not isinstance(parsed, dict):
        raise RegionalTokenError('not_region_map')
    if not parsed:
        raise RegionalTokenError('empty_region_map')

    invalid_regions = sorted(
        region
        for region, token in parsed.items()
        if not isinstance(region, str) or not isinstance(token, str) or not token.strip()
    )
    if invalid_regions:
        raise RegionalTokenError('invalid_region_tokens', regions=invalid_regions)

    unsupported_regions = sorted(set(parsed) - set(supported_regions))
    if unsupported_regions:
        raise RegionalTokenError('unsupported_regions', regions=unsupported_regions)

    return dict(parsed)
