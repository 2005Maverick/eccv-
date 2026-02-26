"""
Question templates with {variable} slots for multilingual translation.
Template-first approach: translate templates once, fill variables after.
"""

from typing import Dict, List

# Templates organized by bias type
# Each template has:
#   - template: the question string with {variable} placeholders
#   - variables: list of variable names that must be filled
#   - answer_format: expected answer format

QUESTION_TEMPLATES: Dict[str, List[Dict]] = {
    "texture": [
        {
            "template": "Do both images show the same type of object? Answer Yes or No.",
            "variables": [],
            "answer_format": "Yes/No",
        },
        {
            "template": "Is the object in Image B a silhouette of the object in Image A? Answer Yes or No.",
            "variables": [],
            "answer_format": "Yes/No",
        },
    ],
    "counting": [
        {
            "template": "Which image has more {object_category}? Answer A or B.",
            "variables": ["object_category"],
            "answer_format": "A/B",
        },
        {
            "template": "How many {object_category} are in Image A? Give a number.",
            "variables": ["object_category"],
            "answer_format": "integer",
        },
        {
            "template": "Count the {object_category} in each image. Which has more?",
            "variables": ["object_category"],
            "answer_format": "A/B",
        },
    ],
    "spatial_relations": [
        {
            "template": "Are these the same scene from the same viewpoint? Answer Yes or No.",
            "variables": [],
            "answer_format": "Yes/No",
        },
        {
            "template": "Has this image been flipped horizontally or vertically? Answer horizontally, vertically, or neither.",
            "variables": [],
            "answer_format": "choice",
        },
    ],
    "physical_plausibility": [
        {
            "template": "Is the lighting in this image physically possible? Answer Yes or No.",
            "variables": [],
            "answer_format": "Yes/No",
        },
        {
            "template": "Are the shadows in this image consistent with a single light source? Answer Yes or No.",
            "variables": [],
            "answer_format": "Yes/No",
        },
    ],
    "temporal_reasoning": [
        {
            "template": "Order these images chronologically. Answer as A,B,C or A,C,B etc.",
            "variables": [],
            "answer_format": "ordering",
        },
        {
            "template": "Which image was taken earliest in the day?",
            "variables": [],
            "answer_format": "A/B/C",
        },
    ],
    "spurious_correlation": [
        {
            "template": "What is the main {object_category} in this image? Name it.",
            "variables": ["object_category"],
            "answer_format": "label",
        },
        {
            "template": "Is this the same type of object in both images, regardless of background? Answer Yes or No.",
            "variables": [],
            "answer_format": "Yes/No",
        },
    ],
    "compositional_binding": [
        {
            "template": "Which description correctly matches the image?",
            "variables": [],
            "answer_format": "multiple_choice",
        },
        {
            "template": "What color is the {object_a} in the image?",
            "variables": ["object_a"],
            "answer_format": "color",
        },
    ],
    "text_in_image": [
        {
            "template": "Are these the same location? Use only the visible text to decide. Answer Yes or No.",
            "variables": [],
            "answer_format": "Yes/No",
        },
        {
            "template": "What text is visible in Image A?",
            "variables": [],
            "answer_format": "text",
        },
    ],
    "scale_invariance": [
        {
            "template": "Do these show the same type of object? Answer Yes or No.",
            "variables": [],
            "answer_format": "Yes/No",
        },
        {
            "template": "Are these objects the same real-world size? Answer Yes or No.",
            "variables": [],
            "answer_format": "Yes/No",
        },
    ],
}
