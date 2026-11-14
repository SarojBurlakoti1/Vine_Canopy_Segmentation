# ImageJ re-measurement: step-by-step protocol

You will measure the canopy area of the 169 test images again in ImageJ, using the same method as in R1. One thing is new: the 4 ft line, the rectangle and the final mask of every image are saved. Then the ground-truth and model areas can be computed inside exactly the same rectangle with exactly the same scale, which is what the reviewer asked for.

The pictures are the dataset images, the same pixels the models and the annotations use. They have been turned upright and given their true shape. Don't edit, rename or move them.

**Time:** about 1.5–2 minutes per image, so roughly 5–6 hours for 169. You can stop and continue at any time.

## Before you start

1. **Don't look at the model results or the ground-truth areas** for these images until everything is measured.
2. Open ImageJ.

## First: practice on 2 images (about 5 minutes)

This tests the macro on the analyst's computer before the real measurements start. The two practice images are not among the 169, and their results are not used.

1. In ImageJ choose **Plugins › Macros › Run…** and select `Measure_Vine_Canopy.ijm` in this folder.
2. When it asks for a folder, choose **`ImageJ_remeasurement/practice`** (one level up from this folder).
3. Do the 4 steps below for both images, exactly as you would for real.
4. The saved files of the practice images were checked before the 169 images were measured.

## Start the macro

1. In ImageJ choose **Plugins › Macros › Run…** and select `Measure_Vine_Canopy.ijm`, which is in this folder.
2. When it asks for a folder, choose **`images_to_measure`**.
3. The macro opens the first image that isn't measured yet. The images are numbered 001–169 in a random order; please keep that order.

## For each image: 4 steps

The macro shows a small window for each step. Do the step, then click **OK** in that window.

**Step 1 – Scale (4 ft line)**
- Draw a straight line from the **centre of one trellis post to the centre of the next post**. That distance is 4 ft = 121.92 cm.
- Draw it at the height of the **middle of the plant**. The posts lean slightly in the photos, so the height matters.
- If a post is hidden, use the outer edges of the 4 ft trellis panel and write that in the notes at the end.
- If there is no 4 ft reference at all, draw any line and write **NO SCALE** in the notes. The image will then be excluded.

**Step 2 – Rectangle**
- Draw **one rectangle that contains the whole plant**, including shoots that grow outside the trellis or past the top of the backdrop.
- Keep other plants out of it where you can, and stay off the gray side bars on the `TRT_` images.

**Step 3 – Threshold**
- The macro now shows only the part of the image inside your rectangle (cropped), so the rest of the photo can't distract you or affect the threshold.
- The Color Threshold window opens. Set it **as you did in R1** (HSB or RGB) so that the plant's **leaves and stems** are selected and the backdrop, shadows and wires are not. Set it separately for every image.
- Click **Select** in the Color Threshold window, then **OK** in the macro window.

**Step 4 – Clean-up**
- A black-and-white mask appears (white = canopy, only inside your rectangle). Compare it with the photo.
- **Remove what is not this plant**, such as background trees, grass, neighbouring vines, shadows and backdrop edges: draw around it with the freehand tool and press **Backspace**.
- **Add canopy that was missed**: draw around it and choose **Edit › Fill**.
- Don't fill the gaps between leaves by hand. The macro runs Fill Holes automatically, as in R1.
- Click **OK**.

The macro then saves everything and asks for **notes**, which are optional. Note anything unusual, for example "post hidden, used trellis edges", "neighbouring vine removed" or "plant partly outside the photo". It then asks whether to continue with the next image.

## Mistakes and breaks

- **Stop:** click "No" when the macro asks whether to continue. To continue later, run the macro again on the same folder; it skips the images already done.
- **Re-do an image:** open `images_to_measure_results/measurements.csv`, delete that image's row, save, and run the macro again.

## When all 169 are done

Tell me. I run an automatic check on every row: mask, rectangle, scale line and area. If a row fails, you re-measure only that image, and I log the reason.

## Repeat measurement (20 images, recommended, about 40 minutes)

This measures how repeatable the ImageJ measurement itself is (scale line, rectangle and threshold). The first round of review asked for that ("quantify analyst, scale, and perspective-calibration uncertainty").

- Do it **on a different day, after all 169 are finished**, without looking at your first results.
- Run the macro and choose the folder **`repeat_20`**. The files are named R01–R20 so you don't know which image is which.
- **Don't open `repeat_key.csv`.**

## Files in this folder

| File | What it is |
|---|---|
| `images_to_measure/` | The 169 images (001–169) |
| `repeat_20/` | 20 of them again, under neutral names |
| `Measure_Vine_Canopy.ijm` | The macro |
| `measurement_manifest.csv` | Which file is which dataset image, and how it was turned upright |
| `repeat_key.csv` | Which R-file is which image (don't open) |
| `images_to_measure_results/`, `repeat_20_results/` | Created by the macro: `measurements.csv`, masks, line and rectangle |
