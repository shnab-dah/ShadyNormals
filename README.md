# Hemisphere Normal Map Hillshade Viewer

**Version:** 0.4 – 2026-05-23  
**Purpose:** interactive visualisation of low-relief heritage objects from RGB normal maps  
**Typical objects:** coins, seals, tablets, worn inscriptions, impressed marks, stamped surfaces, shallow carvings, corrosion-covered relief, and other small-scale surface features.

---

## 1. Overview

The **Hemisphere Normal Map Hillshade Viewer** is a Python application for rendering shaded and enhanced views of low-relief surfaces from an RGB normal map. It is designed for scholarly visual inspection of heritage objects where fine surface orientation changes may be more informative than visible colour or texture alone.

The program takes an image in which the red, green, and blue channels encode per-pixel surface-normal components. It then produces a range of interactive renderings: single-light hillshading, multi-light combinations, slope-like maps, local normal-deviation maps, curvature-like maps, and normal-gradient maps.

The viewer is intended to support interpretation and documentation. It should be understood as a **normal-map visualisation tool**, not as a calibrated metrology tool. It does not reconstruct a metric height map, and it does not estimate absolute object depth.

---

## 2. Scientific context

Reflectance Transformation Imaging (RTI) and multi-light imaging are widely used in cultural heritage because a fixed camera and variable illumination can reveal surface details that are difficult to see under ordinary lighting. RTI viewers allow virtual relighting from different directions, making surface morphology, inscriptions, tool marks, and conservation changes more legible.

This software follows the same interpretative principle, but it works from an already computed **normal map** rather than directly from the original image stack. A normal map stores, for each pixel, the estimated local orientation of the surface. The viewer then uses synthetic lighting and normal-domain enhancement operations to make surface features more visible.

This distinction is important:

- RTI or photometric-stereo processing may estimate surface normals from multiple light directions.
- This viewer assumes that the normal map already exists.
- The viewer renders visual products from those normals.
- The viewer does not itself solve the original photometric-stereo or RTI-fitting problem.
- The viewer does not integrate normals into a height map.

The resulting images are best described as **visual enhancement renders** or **normal-derived visualisations**. They can be extremely useful for reading worn inscriptions or comparing surface morphology, but they should not automatically be treated as measured height data.

---

## 3. Installation

Install Python dependencies:

```bash
pip install numpy pillow pyqtgraph PyQt6
```

Run the viewer:

```bash
python heritage_normal_hillshade_viewer.py
```

or open a normal map directly:

```bash
python heritage_normal_hillshade_viewer.py normal_map.png
```

For very large normal maps, use a smaller interactive preview while preserving full-resolution export:

```bash
python heritage_normal_hillshade_viewer.py normal_map.png --max-preview-size 1200
```

---

## 4. Input data

### 4.1 Required input

The software expects an RGB image in which the channels encode a normal vector:

```text
R = X component of the surface normal
G = Y component of the surface normal
B = Z component of the surface normal
```

The program decodes the RGB values as:

```text
n = 2 * RGB - 1
```

where RGB has first been scaled from `[0, 255]` to `[0, 1]`. Thus:

```text
0   becomes -1
128 becomes approximately 0
255 becomes +1
```

After decoding, each normal is renormalised to unit length:

```text
n = n / ||n||
```

This renormalisation is important because normal maps exported from other software may contain rounding errors or non-unit vectors.

### 4.2 Alpha channel

If the input image has transparency, the alpha channel is preserved as a display mask. Transparent areas are composited onto white in the final render.

### 4.3 Coordinate convention

The viewer assumes the following coordinate system:

```text
X = image right
Y = image down
Z = out of the image, toward the viewer
```

This convention is common in image-processing contexts but may differ from some graphics or photometric-stereo programs. If the relief appears inverted, mirrored, or physically implausible, use the channel-flip checkboxes described below.

---

## 5. Main interface

### 5.1 Image viewer

The central image area shows the current render. It supports:

```text
Mouse wheel = zoom in / zoom out
Left-drag   = pan
Reset zoom  = fit image to window
```

The viewer uses a PyQtGraph image canvas for speed. Internally, the display is inverted on the Y axis so that row 0 appears at the top, matching ordinary image-viewer behaviour.

### 5.2 Load image

The **Load image** button opens an RGB normal map.

When a new image is loaded, the default output filename is generated from the input filename:

```text
input:  coin_normal.png
output: coin_normal_single_light.png, coin_normal_range_multi_light.png, etc.
```

The exact suffix depends on the active render mode.

### 5.3 Save current render

The **Save current render** button exports the current render at full resolution, not merely the preview resolution.

Exported PNG and TIFF files include processing metadata. For other file types, a sidecar metadata JSON file is written.

### 5.4 Save 16 single-light renders

The **Save 16 single-light renders** button exports sixteen ordinary raking-light renders at equally spaced azimuths:

```text
0°, 22.5°, 45°, 67.5°, ..., 337.5°
```

These use the current:

```text
altitude
ambient fill
gamma
normal-channel flip settings
invert-tones setting
```

The active render mode is ignored for this export; the output is always sixteen single-light hillshades.

A contact sheet is also written for rapid comparison.

### 5.5 Invert tones

The **Invert tones** button applies a final tonal inversion:

```text
output = 255 - output
```

It affects both grayscale and RGB renders.

This is applied after the selected render mode has been computed. It does not change the lighting direction, the normal map, or the underlying algorithm. It only reverses black and white tones. This is useful because faint inscriptions or relief edges are sometimes easier to interpret when highlight/shadow polarity is reversed.

### 5.6 Render mode

The **Render mode** dropdown selects the visualisation algorithm.

Available modes:

```text
Single light
Mean multi-light
Max multi-light
Min multi-light
Range multi-light
Std-dev multi-light
RGB 3-light composite
Slope from normals
Local normal deviation
Curvature from normals
Normal gradient magnitude
```

Each mode is described in detail below.

### 5.7 Light direction hemisphere

The hemisphere widget controls synthetic light direction.

Interpretation:

```text
centre of disk = overhead light, altitude 90°
edge of disk   = grazing light, altitude 0°
top            = light from north / top of image
right          = light from east / right of image
bottom         = light from south / bottom of image
left           = light from west / left of image
```

The white dot represents the current light vector. Drag it to change the light direction.

The corresponding azimuth and altitude are displayed below the hemisphere and in the information box.

### 5.8 Ambient fill slider

Range in the current software:

```text
0.00 to 0.80
```

The ambient slider affects light-based modes by adding a constant fill term after Lambertian shading:

```text
I_ambient = A + (1 - A) * I
```

where:

```text
A = ambient fill
I = raw Lambertian intensity
```

Consequences:

- `A = 0.00` gives full black shadows.
- Higher values lift dark regions.
- High ambient values reduce contrast but can make shadowed features easier to inspect.
- Ambient fill is not physically calibrated illumination. It is a display aid.

Ambient fill affects:

```text
Single light
Mean multi-light
Max multi-light
Min multi-light
Range multi-light
Std-dev multi-light
RGB 3-light composite
Save 16 single-light renders
```

Ambient fill does **not** affect:

```text
Slope from normals
Local normal deviation
Curvature from normals
Normal gradient magnitude
```

### 5.9 Gamma / display contrast slider

Range in the current software:

```text
0.30 to 2.50
```

Gamma is applied after the selected render mode has produced a normalised image:

```text
I_gamma = I ^ gamma
```

Consequences:

- `gamma = 1.00` leaves tones unchanged.
- `gamma < 1.00` brightens midtones.
- `gamma > 1.00` darkens midtones.
- Gamma is a display/contrast operation, not a physical lighting parameter.

Gamma affects every render mode.

### 5.10 Multi-light directions slider

Range in the current software:

```text
4 to 64
```

This slider controls the number of evenly spaced azimuthal light directions used in the multi-light modes.

For example:

```text
16 directions = one light every 22.5°
32 directions = one light every 11.25°
```

The lights are distributed around a full 360° circle. Their altitude is taken from the current hemisphere control.

This slider affects only:

```text
Mean multi-light
Max multi-light
Min multi-light
Range multi-light
Std-dev multi-light
```

It does not affect:

```text
Single light
RGB 3-light composite
Slope from normals
Local normal deviation
Curvature from normals
Normal gradient magnitude
```

### 5.11 Local radius slider

Range in the current software:

```text
1 to 80 pixels
```

This slider currently affects only **Local normal deviation**. It controls the size of the local averaging window used to estimate the broad, local surface trend.

The implemented box-filter window size is:

```text
window size = 2 * radius + 1
```

Examples:

```text
radius 5  -> 11 × 11 pixel neighbourhood
radius 12 -> 25 × 25 pixel neighbourhood
radius 40 -> 81 × 81 pixel neighbourhood
```

Small radius values enhance fine scratches and narrow strokes. Large radius values suppress broader curvature and emphasise larger local disturbances.

The local radius slider does not affect the other render modes.

### 5.12 Flip X / Red, Flip Y / Green, Flip Z / Blue

These checkboxes multiply one normal component by `-1`.

They are included because different RTI, photometric-stereo, graphics, and image-processing tools use different normal-map conventions.

Use these controls when:

- inscriptions look like raised relief when they should be incised,
- raised relief looks sunken,
- the light appears to come from the wrong side,
- the image appears mirrored in its shading response.

Typical first test:

```text
Try Flip Y / Green
```

because image Y-axis conventions often differ between software packages.

The effect of these checkboxes depends on render mode. In some modes, flipping a sign has no visible effect because the algorithm uses squared magnitudes or angular differences that are sign-invariant.

---

## 6. Mathematical basis

### 6.1 Normal vector

Each pixel has a normal vector:

```text
n = (nx, ny, nz)
```

After decoding and optional channel flipping, the vector is assumed to have unit length:

```text
||n|| = 1
```

### 6.2 Light vector

The light vector is:

```text
l = (lx, ly, lz)
```

with:

```text
||l|| = 1
```

The hemisphere widget maps the white dot to a vector on the upper hemisphere. Centre gives high `lz`; the disk edge gives `lz ≈ 0`.

### 6.3 Lambertian shading

The single-light render uses a Lambertian dot product:

```text
I = max(0, n · l)
```

Expanded:

```text
I = max(0, nx*lx + ny*ly + nz*lz)
```

This is a simple diffuse-lighting model. It is not a full optical model of the object surface. It ignores:

```text
specular reflection
subsurface scattering
cast shadows
interreflections
material colour
camera response
spatially varying albedo
```

For the purpose of studying low-relief morphology from normal maps, this simplified model is often useful because it isolates surface orientation.

---

## 7. Render modes

## 7.1 Single light

### Purpose

A classical raking-light hillshade from one virtual light direction.

This is the most intuitive mode and is close in spirit to moving a lamp around a coin or inscription.

### Algorithm

For each pixel:

```text
I = max(0, n · l)
I = ambient + (1 - ambient) * I
I = I ^ gamma
```

Then the value is converted to 8-bit grayscale.

### Controls that affect this mode

```text
Hemisphere light direction: yes
Ambient fill: yes
Gamma: yes
Multi-light directions: no
Local radius: no
Flip X/Y/Z: yes
Invert tones: yes
```

### Interpretation

This mode is directional. A feature may be obvious under one light direction and nearly invisible under another. It is excellent for interactive inspection, but a single exported view should not be treated as neutral or exhaustive.

---

## 7.2 Mean multi-light

### Purpose

A directionally averaged hillshade.

This mode reduces dependence on a single raking-light direction. It can provide a more balanced overview of relief.

### Algorithm

The software creates `N` lights evenly spaced around 360°, all at the current hemisphere altitude:

```text
l0, l1, ..., lN-1
```

For each light:

```text
Ii = max(0, n · li)
Ii = ambient + (1 - ambient) * Ii
```

Then:

```text
Imean = mean(I0, I1, ..., IN-1)
```

The result is normalised to `[0, 1]`, gamma-corrected, alpha-composited, and exported.

### Controls that affect this mode

```text
Hemisphere altitude: yes
Hemisphere azimuth: no
Ambient fill: yes
Gamma: yes
Multi-light directions: yes
Local radius: no
Flip X/Y/Z: yes
Invert tones: yes
```

### Interpretation

Mean multi-light is less dramatic than raking light, but it is also less biased by a single illumination direction. Subtle directional marks may become less visible than in Range or Std-dev modes.

---

## 7.3 Max multi-light

### Purpose

Shows the brightest response obtained from any of the sampled light directions.

### Algorithm

The software computes the same multi-light stack as above:

```text
I0, I1, ..., IN-1
```

Then:

```text
Imax = max(I0, I1, ..., IN-1)
```

### Controls that affect this mode

```text
Hemisphere altitude: yes
Hemisphere azimuth: no
Ambient fill: yes
Gamma: yes
Multi-light directions: yes
Local radius: no
Flip X/Y/Z: yes
Invert tones: yes
```

### Interpretation

This mode tends to brighten surfaces that face at least one sampled light direction. It may reduce shadow information and can make relief appear smoother, but it can be useful for reducing directional bias.

---

## 7.4 Min multi-light

### Purpose

Shows the darkest response obtained from any of the sampled light directions.

### Algorithm

```text
Imin = min(I0, I1, ..., IN-1)
```

### Controls that affect this mode

```text
Hemisphere altitude: yes
Hemisphere azimuth: no
Ambient fill: yes
Gamma: yes
Multi-light directions: yes
Local radius: no
Flip X/Y/Z: yes
Invert tones: yes
```

### Interpretation

This mode emphasises pixels that fall into darkness under at least one illumination direction. It can reveal grooves, incisions, and steep local relief, but it may also exaggerate noise or normal-map errors.

---

## 7.5 Range multi-light

### Purpose

Highlights how strongly each pixel changes as light rotates around the object.

This is often useful for worn inscriptions, scratches, and low-relief edges.

### Algorithm

Using the multi-light stack:

```text
Irange = max(I0, I1, ..., IN-1) - min(I0, I1, ..., IN-1)
```

The range image is then robustly normalised using the 1st and 99th percentiles.

### Controls that affect this mode

```text
Hemisphere altitude: yes
Hemisphere azimuth: no
Ambient fill: yes
Gamma: yes
Multi-light directions: yes
Local radius: no
Flip X/Y/Z: yes
Invert tones: yes
```

### Interpretation

Flat areas with little orientation change tend to have low range. Small features whose brightness changes strongly under rotating illumination tend to have high range.

This is not a height-difference map. It is a directional reflectance-response range derived from the normal map.

---

## 7.6 Std-dev multi-light

### Purpose

Highlights pixels whose brightness varies strongly across many illumination directions.

This is similar in aim to Range multi-light but uses standard deviation instead of max-minus-min.

### Algorithm

```text
Istd = standard_deviation(I0, I1, ..., IN-1)
```

The result is robustly normalised using the 1st and 99th percentiles.

### Controls that affect this mode

```text
Hemisphere altitude: yes
Hemisphere azimuth: no
Ambient fill: yes
Gamma: yes
Multi-light directions: yes
Local radius: no
Flip X/Y/Z: yes
Invert tones: yes
```

### Interpretation

Std-dev multi-light is useful for identifying pixels with strong orientation-dependent visibility. It may be slightly less sensitive to single extreme values than Range multi-light.

---

## 7.7 RGB 3-light composite

### Purpose

Creates a colour-coded orientation visualisation using three light directions.

This is not intended as naturalistic rendering. It is an analytical visualisation in which colour differences encode directional lighting response.

### Algorithm

The current hemisphere light is used for the red channel:

```text
R = hillshade at current azimuth
```

Two additional lights are generated:

```text
G = hillshade at current azimuth + 120°
B = hillshade at current azimuth + 240°
```

All three use the current altitude.

### Controls that affect this mode

```text
Hemisphere azimuth: yes
Hemisphere altitude: yes
Ambient fill: yes
Gamma: yes
Multi-light directions: no
Local radius: no
Flip X/Y/Z: yes
Invert tones: yes
```

### Interpretation

Colour indicates directional response. Features with different orientations may appear in different colours. This can be helpful for separating crossing strokes, scratches, or relief edges with different directions.

Do not interpret the colours as material colour. They are synthetic.

---

## 7.8 Slope from normals

### Purpose

Shows direction-independent local relief strength.

This mode is useful when the question is not "where does the shadow fall?" but "where is the surface locally tilted?"

### Algorithm

The software computes:

```text
slope_strength = sqrt(nx² + ny²)
```

For a unit normal, this is equivalent to:

```text
slope_strength = sin(theta)
```

where `theta` is the angle between the surface normal and the viewing axis.

The result is robustly normalised using the 1st and 99th percentiles, gamma-corrected, and exported.

### Controls that affect this mode

```text
Hemisphere light direction: no
Ambient fill: no
Gamma: yes
Multi-light directions: no
Local radius: no
Flip X/Y/Z: no visible effect in ordinary use
Invert tones: yes
```

### Interpretation

Bright pixels indicate surfaces that are tilted relative to the camera. This can reveal relief boundaries and edges independent of lighting direction.

It does not distinguish between slopes facing left, right, up, or down.

---

## 7.9 Local normal deviation

### Purpose

Suppresses broad shape and emphasises local departures from the surrounding surface.

This is particularly useful for faded inscriptions on curved or uneven objects, because it compares each pixel to its local neighbourhood rather than to a global light direction.

### Algorithm

First, the normal map is locally averaged using a box filter with radius `r`:

```text
n_local = normalised_box_blur(n, radius = r)
```

Then the angular deviation between the original normal and the local average normal is computed:

```text
deviation = arccos( clamp(n · n_local, -1, 1) )
```

The deviation image is robustly normalised using the 1st and 99th percentiles, gamma-corrected, and exported.

### Controls that affect this mode

```text
Hemisphere light direction: no
Ambient fill: no
Gamma: yes
Multi-light directions: no
Local radius: yes
Flip X/Y/Z: no visible effect in ordinary use
Invert tones: yes
```

### Interpretation

Small strokes, scratches, corrosion ridges, and shallow incisions can appear strongly if they deviate from the local surface trend.

The local-radius setting is critical:

- small radius = fine detail and noise
- large radius = broader disturbances
- too large a radius may treat real object curvature as "local detail"
- too small a radius may amplify normal-map noise

This mode is not a measurement of depth. It is an angular contrast map.

---

## 7.10 Curvature from normals

### Purpose

Produces a curvature-like enhancement that can make incised and raised details easier to distinguish.

### Algorithm

The software approximates a divergence-like quantity from the X and Y normal components:

```text
curvature_like = d(nx)/dx + d(ny)/dy
```

The derivatives are computed using NumPy's finite-difference gradient. In image coordinates:

```text
x = image columns
y = image rows
```

The output is centred around mid-gray:

```text
mid-gray = zero curvature-like response
bright   = positive response
dark     = negative response
```

The scale is set using the 99th percentile of the absolute response:

```text
scale = percentile_99(abs(curvature_like))
output = 0.5 + 0.5 * clamp(curvature_like / scale, -1, 1)
```

Gamma and final tonal inversion are then applied.

### Controls that affect this mode

```text
Hemisphere light direction: no
Ambient fill: no
Gamma: yes
Multi-light directions: no
Local radius: no
Flip X / Red: yes, changes sign contribution
Flip Y / Green: yes, changes sign contribution
Flip Z / Blue: no
Invert tones: yes
```

### Interpretation

This mode is useful for highlighting local convex/concave changes and stroke edges. However, it should be described carefully:

```text
It is curvature-like, not calibrated geometric curvature.
```

The result depends on normal-map quality, image scale, finite-difference behaviour, and channel orientation. It is best used for visual inspection, not metric curvature analysis.

---

## 7.11 Normal gradient magnitude

### Purpose

Shows the strength of local changes in the normal field.

This acts as an edge or feature-strength map for surface orientation.

### Algorithm

The program computes spatial gradients of all three normal components:

```text
dnx/dx, dnx/dy
dny/dx, dny/dy
dnz/dx, dnz/dy
```

Then it computes:

```text
magnitude = sqrt(
    (dnx/dx)² + (dnx/dy)² +
    (dny/dx)² + (dny/dy)² +
    (dnz/dx)² + (dnz/dy)²
)
```

The result is robustly normalised using the 1st and 99th percentiles, gamma-corrected, and exported.

### Controls that affect this mode

```text
Hemisphere light direction: no
Ambient fill: no
Gamma: yes
Multi-light directions: no
Local radius: no
Flip X/Y/Z: no visible effect in ordinary use
Invert tones: yes
```

### Interpretation

This mode highlights where the surface orientation changes rapidly. It can reveal edges of lettering, scratches, cracks, tool marks, and corrosion boundaries.

It does not indicate whether a feature is raised or incised. It only indicates the magnitude of local normal change.

---

## 8. Control-effect summary table

| Render mode | Hemisphere azimuth | Hemisphere altitude | Ambient | Gamma | Multi-light directions | Local radius | Flip channels | Invert tones |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single light | yes | yes | yes | yes | no | no | yes | yes |
| Mean multi-light | no | yes | yes | yes | yes | no | yes | yes |
| Max multi-light | no | yes | yes | yes | yes | no | yes | yes |
| Min multi-light | no | yes | yes | yes | yes | no | yes | yes |
| Range multi-light | no | yes | yes | yes | yes | no | yes | yes |
| Std-dev multi-light | no | yes | yes | yes | yes | no | yes | yes |
| RGB 3-light composite | yes | yes | yes | yes | no | no | yes | yes |
| Slope from normals | no | no | no | yes | no | no | usually no visible effect | yes |
| Local normal deviation | no | no | no | yes | no | yes | usually no visible effect | yes |
| Curvature from normals | no | no | no | yes | no | no | X/Y affect sign, Z no | yes |
| Normal gradient magnitude | no | no | no | yes | no | no | usually no visible effect | yes |

---

## 9. Export and metadata

### 9.1 Export resolution

Interactive viewing may use a downscaled preview for performance. Export uses the full-resolution normal map.

### 9.2 PNG metadata

For PNG output, the software writes text metadata fields, including:

```text
Software
SoftwareVersion
Processing
RenderMode
Algorithm
SourceNormalMap
ExportedAtUTC
NormalMapConvention
LightVectorX/Y/Z
AzimuthDegrees
AltitudeDegrees
AmbientFill
Gamma
MultiLightDirections
LocalRadiusPixels
FlipX_Red
FlipY_Green
FlipZ_Blue
InvertTones
PreviewWidth/Height
FullWidth/Height
ProcessingJSON
```

`ProcessingJSON` contains a JSON-formatted copy of the processing metadata.

### 9.3 TIFF metadata

For TIFF output, the software writes the processing metadata into the ImageDescription tag and the software name into the Software tag.

### 9.4 Other formats

For formats that may not preserve metadata reliably, the image is saved and a sidecar JSON file is written:

```text
filename.ext.metadata.json
```

---

## 10. Recommended workflows for heritage study

### 10.1 Initial inspection

Start with:

```text
Single light
```

Move the light around the hemisphere. Inspect the object at several grazing-light directions.

Then test:

```text
Flip Y / Green
```

if the relief looks inverted or unnatural.

### 10.2 Faded inscriptions

Useful modes:

```text
Range multi-light
Std-dev multi-light
Local normal deviation
Curvature from normals
Normal gradient magnitude
```

Suggested starting settings:

```text
Multi-light directions: 16 or 32
Local radius: 8–20 pixels
Gamma: 0.7–1.3
Ambient: 0.10–0.25 for light-based modes
```

### 10.3 Coins

Useful modes:

```text
Single light
RGB 3-light composite
Range multi-light
Local normal deviation
Curvature from normals
```

For coins with broad curvature, **Local normal deviation** can be useful because it suppresses broad surface trends.

### 10.4 Scratches and tool marks

Useful modes:

```text
Slope from normals
Normal gradient magnitude
Range multi-light
Std-dev multi-light
```

### 10.5 Publication figures

For academic publication, it is recommended to export multiple modes rather than relying on one visually pleasing render.

A useful figure set might include:

```text
1. Single light, with azimuth/altitude stated
2. Range multi-light, with number of directions stated
3. Local normal deviation, with radius stated
4. Curvature from normals, labelled as curvature-like enhancement
```

Do not describe curvature-like or normal-deviation images as measured topography unless independent validation has been performed.

---

## 11. Limitations and cautions

### 11.1 Not a height map

The software does not compute object height. It operates directly on normals.

A height map would require normal integration, for example solving for a surface `z(x, y)` whose gradients agree with the normal field. That process can introduce drift, boundary artefacts, low-frequency warping, and ambiguity in absolute height scale. This software deliberately avoids that step.

### 11.2 Not calibrated metrology

Unless the input normal map was produced by a calibrated acquisition and processing pipeline, the outputs should not be treated as quantitative measurements.

The renders are influenced by:

```text
normal-estimation accuracy
light calibration in the original RTI/photometric-stereo capture
surface reflectance
specular highlights
shadows
camera noise
image compression
normal-map convention
masking / transparency
chosen display stretch
gamma
ambient fill
```

### 11.3 Albedo is ignored

The viewer does not use the original colour or albedo image. It renders only from normal vectors. This is useful for isolating shape, but it also means that colour, patina, pigment, and material reflectivity are not represented.

### 11.4 Shadows are synthetic and local

The Lambertian shading model computes only local orientation relative to the light. It does not cast shadows from one part of the object onto another. Therefore, it is better described as normal-based shading than as physically complete illumination simulation.

### 11.5 Curvature mode is approximate

The curvature mode computes a simple divergence-like quantity from the normal field. It is useful as an enhancement, but it is not a fully calibrated differential-geometry curvature measurement.

### 11.6 Multi-light modes depend on display normalisation

Range, standard deviation, slope, local normal deviation, and normal-gradient magnitude are contrast-stretched using robust percentile scaling. This improves visibility but means that pixel values are display-normalised, not absolute physical units.

---

## 12. Suggested citation / method description

For a paper, report, or catalogue entry, a transparent methods description could read:

> Normal-map visualisations were generated with the Hemisphere Normal Map Hillshade Viewer, version 0.4. The input RGB normal map was decoded as `R=X`, `G=Y`, `B=Z`, mapped from `[0,255]` to `[-1,+1]`, and normalised per pixel. Single-light renderings used a Lambertian dot product between the decoded normal vector and a user-selected virtual light vector. Additional enhancement modes included multi-directional hillshade statistics, slope strength from normal components, local angular deviation from a box-filtered normal field, finite-difference normal divergence, and normal-gradient magnitude. Exported images include processing metadata recording render mode, light vector, azimuth, altitude, gamma, ambient fill, channel flips, and other relevant parameters.

For a single-light render, include:

```text
light azimuth
light altitude
ambient fill
gamma
normal channel flips
invert-tones setting
```

For multi-light modes, include:

```text
render mode
number of light directions
altitude
ambient fill
gamma
normal channel flips
invert-tones setting
```

For local normal deviation, include:

```text
local radius in pixels
gamma
invert-tones setting
```

For curvature-like maps, include:

```text
finite-difference normal divergence
percentile scaling
gamma
whether tonal inversion was used
```

---

## 13. References and related methods

The software is conceptually related to RTI, photometric stereo, and relief-visualisation approaches used in cultural heritage and archaeological visualisation. The current implementation is intentionally simpler than full RTI or DEM-analysis packages: it focuses on fast inspection of already computed normal maps.

Recommended background sources:

1. Min, J. et al. 2021. *Reflectance transformation imaging for documenting changes through treatment of Joseon dynasty coins*. Heritage Science 9, 105.  
   https://doi.org/10.1186/s40494-021-00584-3

2. American Institute for Conservation Wiki. *Reflectance Transformation Imaging (RTI).*  
   https://www.conservation-wiki.com/wiki/Reflectance_Transformation_Imaging_%28RTI%29

3. Relief Visualization Toolbox Python documentation.  
   https://rvt-py.readthedocs.io/

4. EarthObservation/RVT_py GitHub repository.  
   https://github.com/EarthObservation/RVT_py

5. Zakšek, K., Oštir, K., and Kokalj, Ž. 2011. *Sky-View Factor as a Relief Visualization Technique*. Remote Sensing 3(2), 398–415.  
   https://doi.org/10.3390/rs3020398

---

## 14. Glossary

### Normal map

An image storing surface orientation per pixel, usually as RGB values. In this software, RGB is interpreted as X, Y, and Z normal-vector components.

### Hillshade

A shaded image produced by illuminating a surface from a chosen direction. In this software, hillshade is computed directly from normal vectors.

### Raking light

Low-angle light that emphasises surface relief. In the hemisphere widget, raking light corresponds to placing the white dot near the edge of the disk.

### Ambient fill

A display parameter that lifts dark shadow values. It is not a physically measured ambient illumination term.

### Gamma

A tonal remapping operation. In this software, values are transformed as `I^gamma`.

### Multi-light render

A render computed from many light directions and then combined using a statistic such as mean, maximum, minimum, range, or standard deviation.

### Local normal deviation

An angular contrast image comparing each normal vector to a locally averaged normal vector.

### Curvature-like render

An approximate enhancement based on spatial derivatives of the X and Y normal components. It should not be treated as calibrated curvature.

---

## 15. Development notes

The software is written in Python and uses:

```text
NumPy       numerical array processing
Pillow      image loading, saving, and metadata writing
PyQtGraph   fast interactive image display
PyQt6       user interface
```

The code is designed for readability and transparency rather than maximum computational optimisation. Very large normal maps can still be heavy in modes that compute many full-resolution renders, especially multi-light statistics with high numbers of directions. Use a smaller preview size for interactive work and export full resolution when needed.
