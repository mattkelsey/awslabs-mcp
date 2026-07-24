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

"""Global orchestration for the AWS Compute Optimizer Automation tool."""

import asyncio
import base64
import binascii
import json
from ..utilities.aws_service_base import format_response, handle_aws_error
from ..utilities.sql_utils import convert_response_if_needed
from .compute_optimizer_automation_operations import (
    VALID_OPERATIONS,
    _collect_automation_event_steps,
    _collect_automation_event_summaries,
    _collect_automation_events,
    _collect_automation_rule_preview,
    _collect_automation_rule_preview_summaries,
    _collect_recommended_action_summaries,
    _collect_recommended_actions,
    _format_automation_event,
    create_compute_optimizer_automation_client,
)
from fastmcp import Context
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


# The AWS regions where Compute Optimizer Automation is available. The service is
# absent from botocore's endpoints.json (it ships only an endpoint rule set), so
# there is no local API to enumerate its regions; this list is maintained by hand.
COMPUTE_OPTIMIZER_AUTOMATION_REGIONS = [
    'ap-northeast-1',
    'ap-northeast-2',
    'ap-northeast-3',
    'ap-south-1',
    'ap-southeast-1',
    'ap-southeast-2',
    'ca-central-1',
    'eu-central-1',
    'eu-north-1',
    'eu-west-1',
    'eu-west-2',
    'eu-west-3',
    'sa-east-1',
    'us-east-1',
    'us-east-2',
    'us-west-1',
    'us-west-2',
]

_MAX_CONCURRENT_REGIONS = 8
_SERVICE_NAME = 'Compute Optimizer Automation'


async def dispatch_global(
    ctx: Context,
    operation: str,
    event_id: Optional[str] = None,
    filters: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    rule_type: Optional[str] = None,
    recommended_action_types: Optional[str] = None,
    organization_scope: Optional[str] = None,
    criteria: Optional[str] = None,
    max_results: Optional[int] = None,
    max_pages: int = 10,
    next_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a fan-out operation across all Compute Optimizer Automation regions.

    Handles the operations that carry region-scoped data (events, recommended
    actions, and their summaries and previews). get_automation_event is located by
    ID across regions; the list operations paginate each region and merge.

    Args:
        ctx: The MCP context object.
        operation: The requested region-scoped operation.
        event_id: Automation event ID (get_automation_event, list_automation_event_steps).
        filters: Optional JSON string list of {name, values} filter objects.
        start_time: Optional inclusive start datetime (list_automation_events).
        end_time: Optional exclusive end datetime (list_automation_events).
        start_date: Optional inclusive start date (list_automation_event_summaries).
        end_date: Optional exclusive end date (list_automation_event_summaries).
        rule_type: Rule type for the preview operations.
        recommended_action_types: JSON string array of action types (preview operations).
        organization_scope: Optional JSON string {accountIds: [...]} (preview operations).
        criteria: Optional JSON string of rule criteria (preview operations).
        max_results: Optional maximum number of results per page.
        max_pages: Maximum number of API pages to fetch per region. Defaults to 10.
        next_token: Optional opaque global token to resume regions with more pages.

    Returns:
        The merged multi-region response, or an error response.
    """
    # get_automation_event is a lookup by ID (no pagination) across regions.
    if operation == 'get_automation_event':
        return await get_automation_event_global(ctx, str(event_id))

    regions_tokens, token_error = _parse_global_next_token(next_token)
    if token_error is not None:
        return token_error

    # Each entry: (list_key, collect(client, token) -> (items, token), not_found_is_empty).
    global_handlers = {
        'list_automation_events': (
            'automation_events',
            lambda client, token: _collect_automation_events(
                ctx, client, filters, start_time, end_time, max_results, max_pages, token
            ),
            False,
        ),
        'list_automation_event_steps': (
            'automation_event_steps',
            lambda client, token: _collect_automation_event_steps(
                ctx, client, str(event_id), max_results, max_pages, token
            ),
            True,
        ),
        'list_automation_event_summaries': (
            'automation_event_summaries',
            lambda client, token: _collect_automation_event_summaries(
                ctx, client, filters, start_date, end_date, max_results, max_pages, token
            ),
            False,
        ),
        'list_recommended_actions': (
            'recommended_actions',
            lambda client, token: _collect_recommended_actions(
                ctx, client, filters, max_results, max_pages, token
            ),
            False,
        ),
        'list_recommended_action_summaries': (
            'recommended_action_summaries',
            lambda client, token: _collect_recommended_action_summaries(
                ctx, client, filters, max_results, max_pages, token
            ),
            False,
        ),
        'list_automation_rule_preview': (
            'preview_results',
            lambda client, token: _collect_automation_rule_preview(
                ctx,
                client,
                str(rule_type),
                str(recommended_action_types),
                organization_scope,
                criteria,
                max_results,
                max_pages,
                token,
            ),
            False,
        ),
        'list_automation_rule_preview_summaries': (
            'preview_result_summaries',
            lambda client, token: _collect_automation_rule_preview_summaries(
                ctx,
                client,
                str(rule_type),
                str(recommended_action_types),
                organization_scope,
                criteria,
                max_results,
                max_pages,
                token,
            ),
            False,
        ),
    }

    spec = global_handlers.get(operation)
    if spec is None:
        return format_response(
            'error',
            {'provided_operation': operation, 'valid_operations': VALID_OPERATIONS},
            f'Unsupported operation: {operation}. Valid operations: {", ".join(VALID_OPERATIONS)}.',
        )

    list_key, collect, not_found_is_empty = spec
    return await run_global_list(
        ctx, operation, list_key, regions_tokens, collect, not_found_is_empty
    )


def _encode_global_next_token(region_next_tokens: Dict[str, str]) -> str:
    """Encode per-region pagination state as one opaque tool token."""
    payload = json.dumps(region_next_tokens, separators=(',', ':'), sort_keys=True)
    return base64.b64encode(payload.encode('utf-8')).decode('ascii')


def _parse_global_next_token(
    next_token: Optional[str],
) -> Tuple[Dict[str, Optional[str]], Optional[Dict[str, Any]]]:
    """Resolve a global token into the regions and AWS tokens to resume.

    Global pagination state is an opaque base64-encoded JSON object. Callers must
    return the token unchanged; the regional map is an implementation detail.
    """
    if not next_token:
        return dict.fromkeys(COMPUTE_OPTIMIZER_AUTOMATION_REGIONS), None

    try:
        decoded = base64.b64decode(next_token.encode('ascii'), validate=True).decode('utf-8')
        parsed = json.loads(decoded)
    except (UnicodeEncodeError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError) as e:
        return {}, format_response(
            'error',
            {
                'parameter': 'next_token',
                'supported_regions': COMPUTE_OPTIMIZER_AUTOMATION_REGIONS,
            },
            'Invalid global next_token. If this token came from a global response, pass '
            'it back unchanged. If it came from an explicit-region query, pass `region` '
            f'along with it. Decode error: {e}',
        )

    if not isinstance(parsed, dict):
        return {}, format_response(
            'error',
            {'parameter': 'next_token'},
            'Invalid global next_token: decoded pagination state must be a non-empty '
            'region-to-token map. Pass the previous global response next_token unchanged.',
        )
    if not parsed:
        return {}, format_response(
            'error',
            {'parameter': 'next_token'},
            'Invalid global next_token: the regional pagination map is empty. Start a new '
            'global query by omitting next_token.',
        )

    invalid_tokens = sorted(
        region
        for region, token in parsed.items()
        if not isinstance(region, str) or not isinstance(token, str) or not token.strip()
    )
    if invalid_tokens:
        return {}, format_response(
            'error',
            {'parameter': 'next_token', 'invalid_regions': invalid_tokens},
            'Invalid global next_token: every regional token must be a non-empty string. '
            'Pass the previous global response next_token unchanged.',
        )

    unknown_regions = sorted(set(parsed) - set(COMPUTE_OPTIMIZER_AUTOMATION_REGIONS))
    if unknown_regions:
        return {}, format_response(
            'error',
            {
                'parameter': 'next_token',
                'unsupported_regions': unknown_regions,
                'supported_regions': COMPUTE_OPTIMIZER_AUTOMATION_REGIONS,
            },
            'Invalid global next_token: it contains unsupported region key(s): '
            f'{", ".join(unknown_regions)}. Pass the previous global response next_token '
            'unchanged.',
        )

    return dict(parsed), None


def _is_resource_not_found(error: Exception) -> bool:
    """Return True if the error is a Compute Optimizer Automation not-found error.

    Recognizes both a real botocore ClientError (via its error code) and a bare
    exception class named ResourceNotFoundException.
    """
    response = getattr(error, 'response', None)
    if isinstance(response, dict):
        if response.get('Error', {}).get('Code') == 'ResourceNotFoundException':
            return True
    return type(error).__name__ == 'ResourceNotFoundException'


async def _format_region_error(ctx: Context, error: Exception, operation: str) -> Dict[str, Any]:
    """Classify a regional failure using the shared AWS error handler."""
    classified = await handle_aws_error(ctx, error, operation, _SERVICE_NAME)
    useful_fields = (
        'error_type',
        'message',
        'request_id',
        'http_status',
        'boto_error_type',
        'exception_type',
        'details',
    )
    return {field: classified[field] for field in useful_fields if field in classified}


async def run_global_list(
    ctx: Context,
    operation: str,
    list_key: str,
    regions_tokens: Dict[str, Optional[str]],
    collect: Callable[[Any, Optional[str]], Awaitable[Tuple[List[Dict[str, Any]], Optional[str]]]],
    not_found_is_empty: bool = False,
) -> Dict[str, Any]:
    """Fan a list operation out across regions concurrently and merge the results.

    Args:
        ctx: The MCP context object.
        operation: The tool operation name, used to prefix the SQL table.
        list_key: The response key holding the merged list of items.
        regions_tokens: Map of region -> start token (None to start from the first
            page). Only these regions are queried; pass every Automation region for a
            fresh global call, or a subset to resume specific regions.
        collect: Coroutine (client, start_token) -> (items, leftover_token) that
            paginates one region. Built by the dispatcher to close over the
            operation's parameters.
        not_found_is_empty: When True, a ResourceNotFoundException from a region is
            treated as an empty result rather than a per-region error (used by
            list_automation_event_steps, whose event lives in a single region).

    Returns:
        A format_response dict with the merged items (or the SQL offload sentinel),
        the regions queried, an opaque leftover token, and any typed per-region errors.
        If no region could be searched, an error response with the per-region outcomes.
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REGIONS)

    async def query_region(
        region: str, token: Optional[str]
    ) -> Tuple[
        str,
        List[Dict[str, Any]],
        Optional[str],
        Optional[Dict[str, Any]],
        bool,
    ]:
        async with semaphore:
            try:
                client = await asyncio.to_thread(
                    create_compute_optimizer_automation_client, region
                )
                items, leftover = await collect(client, token)
                return region, items, leftover, None, False
            except Exception as e:
                if not_found_is_empty and _is_resource_not_found(e):
                    return region, [], None, None, True
                error = await _format_region_error(ctx, e, operation)
                return region, [], None, error, False

    results = await asyncio.gather(
        *(query_region(region, token) for region, token in regions_tokens.items())
    )

    merged: List[Dict[str, Any]] = []
    region_next_tokens: Dict[str, str] = {}
    region_errors: Dict[str, Dict[str, Any]] = {}
    regions_not_found: List[str] = []
    successful_regions = 0
    for region, items, leftover, error, not_found in results:
        if not_found:
            regions_not_found.append(region)
            continue
        if error is not None:
            region_errors[region] = error
            continue
        successful_regions += 1
        for item in items:
            # Events and recommended actions already carry the resource's region;
            # summaries and steps do not, so stamp the queried region on them.
            if not item.get('region'):
                item['region'] = region
            merged.append(item)
        if leftover:
            region_next_tokens[region] = leftover

    if not_found_is_empty and regions_not_found and not region_errors and not successful_regions:
        return format_response(
            'error',
            {
                'operation': operation,
                'regions_queried': list(regions_tokens),
                'regions_not_found': regions_not_found,
            },
            f'The requested resource was not found in any of the {len(regions_not_found)} '
            f'region(s) queried for {operation}.',
        )

    if region_errors and not successful_regions:
        data: Dict[str, Any] = {
            'operation': operation,
            'regions_queried': list(regions_tokens),
            'region_errors': region_errors,
        }
        if regions_not_found:
            data['regions_not_found'] = regions_not_found
        if regions_not_found:
            message = (
                f'Could not determine whether the requested resource exists for {operation}: '
                f'{len(region_errors)} region(s) failed and {len(regions_not_found)} returned '
                'not found.'
            )
        else:
            message = f'All {len(region_errors)} region(s) failed for {operation}.'
        return format_response('error', data, message)

    return await _finalize_global_list_response(
        ctx,
        operation,
        list_key,
        merged,
        list(regions_tokens),
        region_next_tokens,
        region_errors,
    )


async def _finalize_global_list_response(
    ctx: Context,
    operation: str,
    list_key: str,
    items: List[Dict[str, Any]],
    regions_queried: List[str],
    region_next_tokens: Dict[str, str],
    region_errors: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a merged multi-region list response, offloading to SQL when large.

    Mirrors the regional response finalizer. The item list is stored first so the
    SQL records converter extracts it (regions_queried is also a list), and the
    opaque token and typed error map are passed as metadata so they survive an
    offload.

    Args:
        ctx: The MCP context object.
        operation: The tool operation name, used to prefix the SQL table.
        list_key: The response key holding the merged list of items.
        items: The merged, region-annotated items.
        regions_queried: The regions that were queried.
        region_next_tokens: Map of region -> leftover token for regions with more data.
        region_errors: Map of region -> structured error for regions that failed.

    Returns:
        A format_response dict, either the inline merged list or the SQL offload sentinel.
    """
    response_data: Dict[str, Any] = {list_key: items, 'count': len(items)}
    response_data['regions_queried'] = regions_queried
    global_next_token = None
    if region_next_tokens:
        global_next_token = _encode_global_next_token(region_next_tokens)
        response_data['next_token'] = global_next_token
    if region_errors:
        response_data['region_errors'] = region_errors

    offload_metadata: Dict[str, Any] = {'regions_queried': regions_queried}
    if global_next_token:
        # Keep the public key stable even when the generic SQL converter also
        # derives its next_page_token pagination envelope.
        offload_metadata['next_token'] = global_next_token
    if region_errors:
        offload_metadata['region_errors'] = region_errors

    response_data = await convert_response_if_needed(
        ctx,
        response_data,
        f'compute_optimizer_automation_{operation}',
        pagination_token_key='next_token',
        **offload_metadata,
    )

    return format_response('success', response_data)


async def get_automation_event_global(ctx: Context, event_id: str) -> Dict[str, Any]:
    """Locate an automation event by ID across all Automation regions.

    Event IDs carry no region, so this fans out get_automation_event concurrently.
    A ResourceNotFoundException in a region is a miss; other errors are recorded
    per region.

    Args:
        ctx: The MCP context object.
        event_id: The ID of the automation event to retrieve.

    Returns:
        The event and the region it was found in, or an error response listing the
        regions searched (and any per-region errors) when no region has it.
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REGIONS)

    async def lookup(
        region: str,
    ) -> Tuple[
        str,
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
        bool,
    ]:
        async with semaphore:
            try:
                client = await asyncio.to_thread(
                    create_compute_optimizer_automation_client, region
                )
                response = await asyncio.to_thread(client.get_automation_event, eventId=event_id)
                return region, response, None, False
            except Exception as e:
                if _is_resource_not_found(e):
                    return region, None, None, True
                error = await _format_region_error(ctx, e, 'get_automation_event')
                return region, None, error, False

    await ctx.info(f'Searching all Automation regions for automation event {event_id}')
    results = await asyncio.gather(
        *(lookup(region) for region in COMPUTE_OPTIMIZER_AUTOMATION_REGIONS)
    )

    region_errors: Dict[str, Dict[str, Any]] = {}
    regions_not_found: List[str] = []
    for region, response, error, not_found in results:
        if not_found:
            regions_not_found.append(region)
            continue
        if error is not None:
            region_errors[region] = error
        elif response is not None:
            data = {
                'automation_event': _format_automation_event(response),
                'found_in_region': region,
            }
            return format_response('success', data)

    data: Dict[str, Any] = {
        'event_id': event_id,
        'regions_queried': list(COMPUTE_OPTIMIZER_AUTOMATION_REGIONS),
    }
    if region_errors:
        data['region_errors'] = region_errors
        data['regions_not_found'] = regions_not_found
        return format_response(
            'error',
            data,
            f'Could not determine whether automation event {event_id} exists because '
            f'{len(region_errors)} of {len(COMPUTE_OPTIMIZER_AUTOMATION_REGIONS)} region(s) '
            'could not be searched. Review region_errors and retry.',
        )
    return format_response(
        'error', data, f'Automation event {event_id} was not found in any region.'
    )


__all__ = [
    'COMPUTE_OPTIMIZER_AUTOMATION_REGIONS',
    '_encode_global_next_token',
    '_parse_global_next_token',
    'dispatch_global',
]
