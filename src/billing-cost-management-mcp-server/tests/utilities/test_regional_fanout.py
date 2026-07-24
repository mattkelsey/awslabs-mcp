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

"""Tests for generic regional fan-out utilities."""

import pytest
from awslabs.billing_cost_management_mcp_server.utilities.regional_fanout import (
    RegionalTokenError,
    decode_regional_next_token,
    encode_regional_next_token,
    fan_out_regional_pages,
    fan_out_regions,
)


async def test_fan_out_regions_collects_ordered_outcomes():
    """Successes, errors, and misses preserve the caller's region order."""

    async def worker(region, state):
        if state == 'miss':
            raise LookupError(region)
        if state == 'error':
            raise RuntimeError(region)
        return state.upper()

    async def format_error(region, error):
        return {'region': region, 'message': str(error)}

    result = await fan_out_regions(
        {'us-east-1': 'ok', 'us-west-2': 'miss', 'eu-west-1': 'error'},
        worker,
        format_error,
        max_concurrency=2,
        is_miss=lambda error: isinstance(error, LookupError),
    )

    assert result.successes == {'us-east-1': 'OK'}
    assert result.misses == ['us-west-2']
    assert result.errors == {'eu-west-1': {'region': 'eu-west-1', 'message': 'eu-west-1'}}


async def test_fan_out_regions_rejects_invalid_concurrency():
    """A zero-sized semaphore is rejected rather than hanging."""

    async def worker(region, state):
        return state

    async def format_error(region, error):
        return str(error)

    with pytest.raises(ValueError, match='at least 1'):
        await fan_out_regions(
            {'us-east-1': None},
            worker,
            format_error,
            max_concurrency=0,
        )


async def test_fan_out_regional_pages_merges_items_tokens_and_outcomes():
    """Regional pages are merged, stamped, and retain errors and misses."""

    async def worker(region, state):
        if state == 'miss':
            raise LookupError(region)
        if state == 'error':
            raise RuntimeError(region)
        if state == 'more':
            return [{'id': 'one', 'region': ''}], 'service-token'
        return [{'id': 'two', 'region': 'source-region'}], None

    async def format_error(region, error):
        return {'region': region, 'message': str(error)}

    result = await fan_out_regional_pages(
        {
            'us-east-1': 'more',
            'us-west-2': 'done',
            'eu-west-1': 'miss',
            'ap-south-1': 'error',
        },
        worker,
        format_error,
        max_concurrency=2,
        is_miss=lambda error: isinstance(error, LookupError),
    )

    assert result.items == [
        {'id': 'one', 'region': 'us-east-1'},
        {'id': 'two', 'region': 'us-west-2'},
    ]
    assert result.next_tokens == {'us-east-1': 'service-token'}
    assert result.successful_regions == ['us-east-1', 'us-west-2']
    assert result.misses == ['eu-west-1']
    assert result.errors == {'ap-south-1': {'region': 'ap-south-1', 'message': 'ap-south-1'}}


def test_regional_next_token_round_trip():
    """Regional token maps round-trip and omission initializes every region."""
    regions = ['us-east-1', 'us-west-2']
    token = encode_regional_next_token({'us-west-2': 'service-token'})

    assert decode_regional_next_token(token, regions) == {'us-west-2': 'service-token'}
    assert decode_regional_next_token(None, regions) == {
        'us-east-1': None,
        'us-west-2': None,
    }


def test_regional_next_token_rejects_unsupported_region():
    """Decoded state cannot resume a region outside the caller's allowlist."""
    token = encode_regional_next_token({'moon-1': 'service-token'})

    with pytest.raises(RegionalTokenError) as exc_info:
        decode_regional_next_token(token, ['us-east-1'])

    assert exc_info.value.reason == 'unsupported_regions'
    assert exc_info.value.details['regions'] == ['moon-1']
