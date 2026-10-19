# Dreamscapes style bible

The paragraph under "Style block" is pasted word for word at the top of every image
generation prompt, before the scene description and the bridge-object instruction. It is
hashed into the generation log, so any edit to it shows up as style drift between scenes.
Everything outside that paragraph is guidance for people and is never sent to a generator.

## Style block

A still, dreamlike scene in soft painterly realism, like a cinematic matte painting with gentle brushwork. Diffused light through a faint haze, long soft shadows, thin mist softening the distance. Muted palette of dusty teal, slate blue and lavender, with a warm amber glow on every light source. Gentle bloom around lights, fine film grain, slightly lifted blacks, a subtle vignette. Wide establishing shot from eye level, 35 mm lens, deep depth of field. The place is deserted and quiet, as if everyone has just left. No people, figures or faces anywhere, and every surface is plain and unmarked, with no lettering, signs, logos or watermarks. Calm, melancholic, inviting.

## Do not

- Do not edit the style block for a single scene. It must stay identical across all ten so the
  scenes read as one set; scene-specific detail belongs in that scene's `scene` text in
  `content/scenes.yaml`.
- Do not put markdown, quotes, bullet points or line breaks inside the style block; the pipeline
  copies it verbatim.
- Do not add quality boilerplate such as "masterpiece", "trending", "award-winning", "8k" or
  "ultra detailed". It pulls the image toward a glossy, generic look.
- Do not request text of any kind: no signs, posters, labels, book titles, ship names, number
  plates or writing on objects. Generators garble it, and it breaks the dream.
- Do not request people, silhouettes, statues, mannequins, portraits or anything with a face.
  The scenes are empty on purpose.
- Do not name brands, logos, real landmarks or living artists.
- Do not describe the bridge object as clickable, hidden, special or highlighted, in prompts or
  in alt text. It should sit in the scene as naturally as every other object.
- Do not ask the generator for the colour grade itself (exact saturation, grain strength,
  vignette amount). The prompt aims for the look; the pipeline's deterministic grade makes it
  consistent.

## Grade notes

Every selected master is graded by the pipeline after generation, with the parameters in
`content/grade.toml`: desaturation, a black lift with a soft highlight shoulder, a teal and amber
split-tone, bloom on bright areas, a soft vignette and fine luminance grain. That grade is what
makes ten separately generated images share one palette, glow and texture, so tune the look
there, once, rather than in the prompts. The style anchor (scene 1) is attached to later
generations ungraded, so the grade is never applied twice.
