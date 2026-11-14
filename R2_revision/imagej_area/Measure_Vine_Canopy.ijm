// Measure_Vine_Canopy.ijm   version 1.1 (2026-11-11; 1.1: threshold and clean-up on the image cropped to the rectangle)
// ---------------------------------------------------------------------------------------------
// Canopy-area re-measurement for JAFR-D-26-02492, second revision (reviewer comment 3).
// Same ImageJ method as the first revision (4-ft post-to-post scale, manual colour threshold,
// clean-up, Fill Holes, Analyze Particles), but the scale line, the region (rectangle) and the
// final mask of EVERY image are saved, so that ground-truth and model areas can be computed
// inside the same region with the same scale.
//
// HOW TO RUN: in ImageJ choose Plugins > Macros > Run... and select this file.
// Choose the folder "images_to_measure" (or "repeat_20"). The macro opens the first image that is
// not yet in measurements.csv and guides you through 4 steps. After each image it asks whether
// to continue, so you can stop at any time and resume later where you left off.
//
// OUTPUT (next to the chosen folder, in "<folder>_results"):
//   measurements.csv            one row per image (scale, region, pixel count, area in cm2)
//   masks/<image>_mask.png      final canopy mask (white = canopy), same size as the image
//   rois/<image>_rois.zip       scale line and region rectangle (open with the ROI Manager)
// To re-do an image: delete its row from measurements.csv and run the macro again.
// ---------------------------------------------------------------------------------------------

var KNOWN_CM = 121.92;   // distance between adjacent trellis posts: 4 ft
var MIN_PX = 3;          // Analyze Particles minimum size in pixels (R1 used 100 px on photos ~6.3x larger in width)
var VERSION = "1.1";
var TEST = false;
var tl = newArray(0);
var tr = newArray(0);

arg = getArgument();
TEST = startsWith(arg, "TEST;");
if (TEST) {
    p = split(arg, ";");
    dir = p[1];
    tl = split(p[2], ","); tr = split(p[3], ",");
} else {
    dir = getDirectory("Choose the folder with the images to measure");
}
if (!endsWith(dir, "/") && !endsWith(dir, "\\")) dir = dir + File.separator;

out = substring(dir, 0, lengthOf(dir) - 1) + "_results" + File.separator;
File.makeDirectory(out);
File.makeDirectory(out + "masks");
File.makeDirectory(out + "rois");
csv = out + "measurements.csv";
header = "file,image_w,image_h,line_x1,line_y1,line_x2,line_y2,line_px,px_per_cm,roi_x,roi_y,roi_w,roi_h,canopy_px,canopy_area_cm2,min_particle_px,date_time,imagej_version,macro_version,notes";
if (!File.exists(csv)) File.saveString(header + "\n", csv);

run("Options...", "iterations=1 count=1 black");        // binary images: white objects on black
run("Set Measurements...", "area display redirect=None decimal=4");

list = getFileList(dir);
Array.sort(list);
nDone = 0; nTodo = 0;
for (i = 0; i < list.length; i++) {
    if (!isImageFile(list[i])) continue;
    if (alreadyDone(csv, list[i])) nDone++; else nTodo++;
}
if (!TEST) showMessage("Canopy re-measurement", "" + nDone + " images already measured, " + nTodo + " to go.\nResults: " + out);

for (i = 0; i < list.length; i++) {
    name = list[i];
    if (!isImageFile(name)) continue;
    if (alreadyDone(csv, name)) continue;
    measureOne(dir, name, out, csv);
    if (!TEST) {
        if (!getBoolean("Saved " + name + ".\n \nMeasure the next image?")) exit("Stopped. Run the macro again to continue where you left off.");
    }
}
if (!TEST) showMessage("All images in this folder are measured.");

// ---------------------------------------------------------------------------------------------
function isImageFile(f) {
    g = toLowerCase(f);
    r = false;
    if (endsWith(g, ".png") || endsWith(g, ".jpg") || endsWith(g, ".jpeg") || endsWith(g, ".tif")) r = true;
    if (startsWith(f, ".")) r = false;
    return r;
}

function alreadyDone(csvPath, f) {
    s = File.openAsString(csvPath);
    k = indexOf(s, "\n" + f + ",");
    r = false;
    if (k >= 0) r = true;
    return r;
}

function measureOne(dir, name, out, csv) {
    close("*");
    roiManager("reset");
    open(dir + name);
    id = getImageID();
    base = File.nameWithoutExtension;
    W = getWidth(); H = getHeight();
    run("Set Scale...", "distance=0 known=0 unit=pixel");   // start without any scale

    // ---- Step 1: scale line -------------------------------------------------------------
    ok = false;
    while (!ok) {
        if (!TEST) setTool("line");
        if (TEST) makeLine(parseFloat(tl[0]), parseFloat(tl[1]), parseFloat(tl[2]), parseFloat(tl[3]));
        else waitForUser("Step 1 of 4: scale (" + name + ")",
            "Draw a straight line from the CENTRE of one trellis post to the CENTRE of the\n" +
            "next post (4 ft = 121.92 cm), at the height of the MIDDLE of the plant.\n" +
            "If a post is hidden, use the outer edges of the 4-ft trellis panel and say so in the notes.\n \n" +
            "Then click OK.");
        selectImage(id);
        ok = (selectionType() == 5);
        if (!ok) showMessage("Please draw ONE straight line with the Straight Line tool.");
    }
    getLine(x1, y1, x2, y2, lw);
    len = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    ppc = len / KNOWN_CM;
    roiManager("add"); roiManager("select", roiManager("count") - 1); roiManager("rename", "scale_line");
    run("Set Scale...", "distance=" + len + " known=" + KNOWN_CM + " unit=cm");
    run("Select None");

    // ---- Step 2: region (rectangle) -----------------------------------------------------
    ok = false;
    while (!ok) {
        if (!TEST) setTool("rectangle");
        if (TEST) makeRectangle(parseInt(tr[0]), parseInt(tr[1]), parseInt(tr[2]), parseInt(tr[3]));
        else waitForUser("Step 2 of 4: region (" + name + ")",
            "Draw ONE rectangle that contains the WHOLE plant: all leaves and shoots,\n" +
            "also those growing outside the trellis or the backdrop.\n" +
            "Keep other plants and the gray side bars out of it where you can.\n \n" +
            "Then click OK.");
        selectImage(id);
        ok = (selectionType() == 0);
        if (!ok) showMessage("Please draw ONE rectangle with the Rectangle tool.");
    }
    getSelectionBounds(rx, ry, rw, rh);
    rx1 = minOf(W, rx + rw); ry1 = minOf(H, ry + rh);
    rx = maxOf(0, rx); ry = maxOf(0, ry); rw = rx1 - rx; rh = ry1 - ry;
    makeRectangle(rx, ry, rw, rh);
    roiManager("add"); roiManager("select", roiManager("count") - 1); roiManager("rename", "region");
    run("Select None");
    roiManager("Show None");

    // Crop to the rectangle: the threshold and the clean-up see only the region (as the R1 crop).
    selectImage(id);
    makeRectangle(rx, ry, rw, rh);
    run("Duplicate...", "title=[region_" + base + "]");
    cid = getImageID();
    run("Select None");
    selectImage(id); close();

    // ---- Step 3: colour threshold (on the cropped region) ---------------------------------
    ok = false;
    while (!ok) {
        selectImage(cid);
        if (TEST) testThreshold(cid);
        else {
            run("Color Threshold...");
            waitForUser("Step 3 of 4: threshold (" + name + ")",
                "The image now shows only your rectangle.\n" +
                "In the Color Threshold window (HSB or RGB, as in R1), adjust the sliders so that the\n" +
                "plant's leaves and stems are selected and the backdrop, shadows and wires are not.\n" +
                "Then click 'Select' in the Color Threshold window, and click OK here.");
        }
        selectImage(cid);
        ok = (selectionType() != -1);
        if (!ok) showMessage("No selection found. Click 'Select' in the Color Threshold window before OK.\n(If you clicked 'Filtered', click 'Original' first.)");
    }
    if (isOpen("Threshold Color")) { selectWindow("Threshold Color"); run("Close"); selectImage(cid); }
    if (isOpen("Mask")) { selectWindow("Mask"); close(); }
    selectImage(cid);
    run("Create Mask");
    mid = getImageID();
    setBackgroundColor(0, 0, 0); setForegroundColor(255, 255, 255);
    run("Select None");

    // ---- Step 4: clean-up -------------------------------------------------------------------
    if (!TEST) {
        selectImage(mid);
        if (!TEST) setTool("freehand");
        setBackgroundColor(0, 0, 0); setForegroundColor(255, 255, 255);
        waitForUser("Step 4 of 4: clean-up (" + name + ")",
            "This is the canopy mask (WHITE = canopy) of your rectangle.\n" +
            "Compare it with the photo. Remove anything that is not this plant\n" +
            "(other plants, grass, wires, shadows, backdrop edges): draw around it and press Backspace.\n" +
            "Add canopy that was missed: draw around it and choose Edit > Fill.\n \n" +
            "When the mask is right, click OK.");
    }
    selectImage(mid);
    run("Select None");
    if (!is("binary")) exit("The mask is no longer black-and-white (" + name + "). Re-do this image.");
    run("Fill Holes");
    run("Analyze Particles...", "size=" + MIN_PX + "-Infinity pixel show=Masks clear");
    fid = getImageID();
    if (fid == mid) exit("Analyze Particles did not create a mask (" + name + ").");
    if (is("Inverting LUT")) run("Invert LUT");
    // Put the region mask back at its place in a full-size image (black = no canopy outside the rectangle).
    setBackgroundColor(0, 0, 0);
    run("Canvas Size...", "width=" + (rx + rw) + " height=" + (ry + rh) + " position=Bottom-Right zero");
    run("Canvas Size...", "width=" + W + " height=" + H + " position=Top-Left zero");
    if (getWidth() != W || getHeight() != H) exit("Mask size error (" + name + "). Re-do this image.");
    getHistogram(values, counts, 256);
    n = counts[255];
    if (counts[0] + counts[255] != W * H) exit("Final mask is not binary (" + name + ").");
    area = n / (ppc * ppc);
    saveAs("PNG", out + "masks" + File.separator + base + "_mask.png");
    roiManager("deselect");
    roiManager("save", out + "rois" + File.separator + base + "_rois.zip");

    notes = "";
    if (!TEST) notes = getString("Notes for " + name + " (optional), e.g. 'post hidden, used trellis edges':", "");
    notes = replace(replace(notes, ",", ";"), "\n", " ");
    getDateAndTime(year, month, dow, day, hour, minute, second, msec);
    stamp = "" + year + "-" + IJ.pad(month + 1, 2) + "-" + IJ.pad(day, 2) + " " + IJ.pad(hour, 2) + ":" + IJ.pad(minute, 2);
    row = name + "," + W + "," + H + "," + d2s(x1, 2) + "," + d2s(y1, 2) + "," + d2s(x2, 2) + "," + d2s(y2, 2) + "," +
          d2s(len, 3) + "," + d2s(ppc, 6) + "," + rx + "," + ry + "," + rw + "," + rh + "," + n + "," + d2s(area, 3) + "," +
          MIN_PX + "," + stamp + "," + getVersion() + "," + VERSION + "," + notes;
    File.append(row, csv);
    close("*");
}

// Used only by the automatic test (TEST mode): a fixed HSB threshold instead of the interactive one.
function testThreshold(id) {
    selectImage(id);
    run("Duplicate...", "title=tt");
    run("HSB Stack");
    setSlice(1); run("Duplicate...", "title=Hue");
    selectWindow("tt"); setSlice(2); run("Duplicate...", "title=Saturation");
    selectWindow("tt"); setSlice(3); run("Duplicate...", "title=Brightness");
    selectWindow("Hue");        setThreshold(30, 130); run("Convert to Mask");
    selectWindow("Saturation"); setThreshold(60, 255); run("Convert to Mask");
    selectWindow("Brightness"); setThreshold(40, 255); run("Convert to Mask");
    imageCalculator("AND create", "Hue", "Saturation");       rename("hs");
    imageCalculator("AND create", "hs", "Brightness");        rename("hsb");
    run("Create Selection");
    selectImage(id);
    run("Restore Selection");
    selectWindow("tt"); close();
    selectWindow("Hue"); close(); selectWindow("Saturation"); close(); selectWindow("Brightness"); close();
    selectWindow("hs"); close(); selectWindow("hsb"); close();
    selectImage(id);
}
