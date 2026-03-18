"""gRPC server wiring for the Sprint 08 VAD service."""

from __future__ import annotations

import grpc

from .config import VadServiceSettings
from .proto import vad_pb2, vad_pb2_grpc
from .runtime import VadServiceRuntime


class VadAnalyzerService(vad_pb2_grpc.VadAnalyzerServicer):
    """Stream audio frames through the Sprint 08 acoustic segmenter."""

    def __init__(self, runtime: VadServiceRuntime) -> None:
        self.runtime = runtime

    async def AnalyzeStream(self, request_iterator, context):  # noqa: N802 - proto contract
        segmenter = self.runtime.create_segmenter()
        async for frame in request_iterator:
            try:
                result = self.runtime.analyze_frame(
                    segmenter=segmenter,
                    sequence=frame.sequence,
                    sample_rate=frame.sample_rate,
                    pcm16=frame.pcm16,
                )
            except Exception as exc:  # noqa: BLE001 - translate service failures to gRPC status
                await context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))

            yield vad_pb2.VadEvent(
                session_id=frame.session_id,
                state=result.state.value,
                probability=result.probability,
                rms=result.rms,
                segment_id=result.segment_id or "",
                sequence=result.sequence,
                end_of_segment=result.end_of_segment,
            )


async def create_grpc_server(
    settings: VadServiceSettings,
    runtime: VadServiceRuntime,
) -> grpc.aio.Server:
    """Create a gRPC server bound to the configured host and port."""
    server = grpc.aio.server()
    vad_pb2_grpc.add_VadAnalyzerServicer_to_server(VadAnalyzerService(runtime), server)
    server.add_insecure_port(f"{settings.grpc_host}:{settings.grpc_port}")
    return server
