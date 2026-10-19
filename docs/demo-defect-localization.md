# Demo defect localization

AnomalyVFM demo streams display red **AI REGION** boxes from the model's
96 x 96 pixel anomaly map when the image classifier predicts DEFECT.
Dataset labels and ground-truth masks are never used to construct the boxes.
This is a display heuristic for model hotspots, not calibrated segmentation or
proof of the physical defect boundary. Existing image scores and thresholds
are unchanged. A DEFECT prediction with diffuse/flat pixel scores may have no box.

The threshold is the interior median plus 60% of the remaining range to the
maximum. Four-connected components smaller than four pixels are removed;
one pixel of padding is added, and the five strongest regions are shown.
Near-flat maps and regions covering over 80% of the frame are suppressed.
Coordinates are normalized after the same full-frame resize used by the model.

Boxes are tied to the sampled demo image ID, index and loop and expire after
three seconds. The renderer checks the outgoing frame PTS, so a late inference
cannot annotate the next slide. Normal results, inference failures and loop
resets clear boxes. At one slide per second a box appears only after that
slide's inference finishes; pending slides intentionally have no box.

The stream and clip branches receive the same overlay. Model input stays on
the clean appsink branch. Production mode and other backends are unchanged.
The target requires GStreamer cairooverlay and python3-gi-cairo (included in
Debian package dependencies).
