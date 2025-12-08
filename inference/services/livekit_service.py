import asyncio
from typing import Dict, Optional
import numpy as np
import cv2
from livekit import rtc, api

from inference.services.inference_service import InferenceService
from inference.settings import LiveKitConfig, VideoConfig


class ParticipantProcessor:
    """Processes video from a single participant"""

    def __init__(
        self,
        participant: rtc.RemoteParticipant,
        inference_service: InferenceService,
        video_config: VideoConfig,
        room: rtc.Room
    ):
        self.participant = participant
        self.inference_service = inference_service
        self.video_config = video_config
        self.room = room
        self.video_source: Optional[rtc.VideoSource] = None
        self.video_track: Optional[rtc.LocalVideoTrack] = None
        self.is_running = False

    async def start(self, video_track: rtc.RemoteVideoTrack) -> None:
        """Start processing video from the participant"""
        if self.is_running:
            return

        print(f"Starting processing for participant: {self.participant.identity}")
        self.is_running = True

        # Create video source and track for publishing annotated frames
        self.video_source = rtc.VideoSource(
            width=self.video_config.width,
            height=self.video_config.height
        )
        self.video_track = rtc.LocalVideoTrack.create_video_track(
            f"{self.participant.identity}_annotated",
            self.video_source
        )

        # Publish the annotated track
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_CAMERA
        await self.room.local_participant.publish_track(self.video_track, options)
        print(f"Published annotated track for {self.participant.identity}")

        # Start processing frames
        asyncio.create_task(self._process_frames(video_track))

    async def _process_frames(self, video_track: rtc.RemoteVideoTrack) -> None:
        """Process frames from the video track"""
        video_stream = rtc.VideoStream(video_track)

        frame_count = 0
        async for frame_event in video_stream:
            if not self.is_running:
                break

            try:
                # Convert frame to numpy array
                frame = self._frame_to_numpy(frame_event.frame)

                if frame is None:
                    continue

                # Run inference
                result = self.inference_service.infer(frame)

                # Resize annotated frame if needed
                if result.annotated_frame is not None:
                    annotated = cv2.resize(
                        result.annotated_frame,
                        (self.video_config.width, self.video_config.height)
                    )

                    # Convert to RGBA for LiveKit
                    frame_rgba = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGBA)

                    # Create VideoFrame and capture to source
                    lk_frame = rtc.VideoFrame(
                        width=self.video_config.width,
                        height=self.video_config.height,
                        type=rtc.VideoBufferType.RGBA,
                        data=frame_rgba.tobytes()
                    )
                    self.video_source.capture_frame(lk_frame)

                frame_count += 1
                if frame_count % 30 == 0:
                    print(
                        f"[{self.participant.identity}] Processed {frame_count} frames. "
                        f"Last inference: {result.inference_time_ms:.2f}ms, "
                        f"Detections: {len(result.bounding_boxes)}"
                    )

            except Exception as e:
                print(f"Error processing frame for {self.participant.identity}: {e}")
                continue

    def _frame_to_numpy(self, frame: rtc.VideoFrame) -> Optional[np.ndarray]:
        """Convert LiveKit VideoFrame to numpy array"""
        try:
            # Get frame buffer
            buffer = frame.data

            # Convert to numpy array based on format
            if frame.type == rtc.VideoBufferType.RGBA:
                arr = np.frombuffer(buffer, dtype=np.uint8).reshape(
                    (frame.height, frame.width, 4)
                )
                # Convert RGBA to BGR for OpenCV
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif frame.type == rtc.VideoBufferType.RGB:
                arr = np.frombuffer(buffer, dtype=np.uint8).reshape(
                    (frame.height, frame.width, 3)
                )
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                print(f"Unsupported frame type: {frame.type}")
                return None

        except Exception as e:
            print(f"Error converting frame to numpy: {e}")
            return None

    async def stop(self) -> None:
        """Stop processing"""
        self.is_running = False
        if self.video_track:
            await self.room.local_participant.unpublish_track(self.video_track)


class LiveKitService:
    """Service for managing LiveKit room connection and participant processing"""

    def __init__(
        self,
        livekit_config: LiveKitConfig,
        inference_service: InferenceService,
        video_config: VideoConfig
    ):
        self.config = livekit_config
        self.inference_service = inference_service
        self.video_config = video_config
        self.room: Optional[rtc.Room] = None
        self.participant_processors: Dict[str, ParticipantProcessor] = {}

    async def connect(self) -> None:
        """Connect to the LiveKit room"""
        print(f"Connecting to LiveKit room: {self.config.room_name}")

        # Generate access token
        token = api.AccessToken(self.config.api_key, self.config.api_secret)
        token.with_identity("inference-server").with_name("Inference Server")
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=self.config.room_name,
        ))

        # Create and connect room
        self.room = rtc.Room()

        # Set up event handlers
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)

        await self.room.connect(self.config.url, token.to_jwt())
        print(f"Connected to room: {self.config.room_name}")

        # Process existing participants
        for participant in self.room.remote_participants.values():
            await self._on_participant_connected(participant)

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        """Handle new participant connection"""
        print(f"Participant connected: {participant.identity}")

        # Process existing tracks
        for publication in participant.track_publications.values():
            if publication.track is not None:
                asyncio.create_task(self._on_track_subscribed(
                    publication.track,
                    publication,
                    participant
                ))

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        """Handle participant disconnection"""
        print(f"Participant disconnected: {participant.identity}")
        if participant.identity in self.participant_processors:
            processor = self.participant_processors.pop(participant.identity)
            asyncio.create_task(processor.stop())

    async def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant
    ) -> None:
        """Handle new track subscription"""
        print(f"Track subscribed: {track.kind} from {participant.identity}")

        if track.kind == rtc.TrackKind.KIND_VIDEO:
            # Create processor for this participant if not exists
            if participant.identity not in self.participant_processors:
                processor = ParticipantProcessor(
                    participant,
                    self.inference_service,
                    self.video_config,
                    self.room
                )
                self.participant_processors[participant.identity] = processor

            # Start processing this video track
            await self.participant_processors[participant.identity].start(track)

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant
    ) -> None:
        """Handle track unsubscription"""
        print(f"Track unsubscribed: {track.kind} from {participant.identity}")

    async def disconnect(self) -> None:
        """Disconnect from the room"""
        if self.room:
            # Stop all processors
            for processor in self.participant_processors.values():
                await processor.stop()
            self.participant_processors.clear()

            await self.room.disconnect()
            print("Disconnected from room")
