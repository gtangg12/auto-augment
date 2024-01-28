### fast prompt but less diverse
TACTIC_GENERATION_PROMPT = """You are an intuitive human being who spend some time driving. What other surrounding environments might this image occur in? Give qualitative transformations corresponding to real life situations. IMPORTANT: Do not add or remove objects (such as cars or pedestrians), change their locations, or change how busy it is. Focus on environmental changes which do not affect the segmentation mask. Be detailed, for example, "cover road with snow" is better than "make it snowy."
Respond with specific prompts for a robot pix2pix diffusion model for data augmentation. Be imperative and concise, with no more than 5 words per action. Respond in list format, with actions only:
1. ...
2. ...
3. ...
Respond with %s diverse actions.
"""

### slower prompt pair but more diverse
TACTIC_IDEATION_PROMPT = """You are an intuitive human being who spend some time driving. What other surrounding environments might this image occur in? Give qualitative transformations corresponding to real life situations. IMPORTANT: Do not add or remove objects (such as cars or pedestrians), change their locations, or change how busy it is. Focus on environmental changes which do not affect the segmentation mask."""

TACTIC_FORMALIZATION_PROMPT = """Convert these suggestions to specific prompts for a pix2pix diffusion model for data augmentation. Be imperative and concise, with no more than 5 words per action. Each of these augmentation may require multiple actions.

%s

Respond in list format, with actions only:
1. ...
2. ...
3. ...
"""
