# Teacher dataset background execution notes

## Recommended operation

Run `SDXLTeacherDataset` in the foreground while the iPhone is connected to power.
Enable:

- Require charging
- Pause on high thermal state
- Keep screen awake
- Dim progress screen

This is the primary supported mode for 2,000 to 5,000 image teacher runs.

## Checkpoint and resume

The app treats checkpoint/resume as the reliability boundary:

- `manifest.json` is rewritten after every completed job.
- `metadata.jsonl` is appended and flushed after every completed job.
- `failed_jobs.jsonl` records per-job failures.
- On resume, jobs already present in metadata and image folders are skipped.
- If the app is killed or the device reboots, use `Resume Latest`.

## Background and lock behavior

iOS does not generally allow arbitrary long-running ML inference after an app is backgrounded or the device is locked. The app therefore uses `UIApplication.beginBackgroundTask` only as a finite grace period to finish or checkpoint the current job. If the system expires the background task, the app requests cancellation and the run remains resumable.

Do not rely on lock-screen execution for hours. If background execution fails, the dataset should remain consistent because completed jobs are flushed one by one.

## BGContinuedProcessingTask

Apple documents iOS/iPadOS 26 `BGContinuedProcessingTask` as a user-initiated continuous background task for long work that may continue after the app leaves the foreground, including Core ML processing on supported devices. It still has constraints: it must have a defined endpoint, needs task registration and permitted identifiers, can be expired or terminated by the system, and must handle cancellation.

This project prepares permitted identifiers in `SDXLTeacherDataset/Info.plist`, but the current implementation does not depend on `BGContinuedProcessingTask`. The fallback remains foreground execution plus checkpoint/resume. A future iOS 26-only path can wrap a selected run preset in a continued-processing task while preserving the same manifest and metadata checkpoint behavior.

Source checked: Apple Developer Documentation, "Performing long-running tasks on iOS and iPadOS".

## What not to do

- Do not use fake audio, location, or VoIP background modes.
- Do not design the dataset format around guaranteed lock-screen execution.
- Do not buffer many generated images in memory.
- Do not overwrite existing run folders.
