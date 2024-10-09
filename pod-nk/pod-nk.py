import os
import json
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import ell
from utility import save_strings_to_files

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize ell library with configuration
ell.init(store='./logdir', autocommit=True, verbose=True)

# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)

model_name = config['model_name']
topics = config['topics']
n_of_outlines = config['n_of_outlines']
n_of_script_drafts = config['n_of_script_drafts']
concept_sys_prompt = config['concept_sys_prompt']
outline_sys_prompt = config['outline_sys_prompt']
eval_outline_sys_prompt = config['eval_outline_sys_prompt']
script_sys_prompt = config['script_sys_prompt']
eval_script_sys_prompt = config['eval_script_sys_prompt']
desc_sys_prompt = config['desc_sys_prompt']
art_sys_prompt = config['art_sys_prompt']
song_sys_prompt = config['song_sys_prompt']
episode_sys_prompt = config['episode_sys_prompt']
selected_topic_index = config['selected_topic_index']

selected_topic = topics[selected_topic_index]
directory = f"/workspaces/pod-scripter/output-nk/{selected_topic}"

@ell.simple(model=model_name, client=OpenAI(api_key=openai_api_key),  temperature=1.0)
def generate_episode_concept(topic : str):
    concept_user_prompt = f"""Create a compelling episode concept for 'Neural Kaleidoscope' based on the theme: {topic}. The concept should be innovative, slightly provocative, and offer practical insights for personal growth. Respond using the following format: 
        Title: [A creative, attention-grabbing title], 
        Description: [A single sentence that encapsulates the episode's core idea and its potential impact on listeners' lives]"""
    return [
        ell.system(concept_sys_prompt),
        ell.user(concept_user_prompt)
    ]

@ell.simple(model=model_name, client=OpenAI(api_key=openai_api_key), temperature=1.0)
def generate_episode_outline(concept : str):
    outline_user_prompt = f"""Given the concept: {concept}, create a flash card-style outline for a short (< 15 min) 'NeuralKaleidoscope' episode.
       Rewrite the title and description in a single line, making it more compelling if needed.
       Generate exactly 5 bullet points that are:
       Highly thought-provoking
       Perspective-changing
       Concise yet impactful
       Based on research or established theories (no expert interviews)
       Practical or insightful for listeners
       Mind-bending analogies & interesting parallels drawn
    Exclude introductions, conclusions, and time allocations. Focus solely on the core ideas that will drive the episode's content."""
    return [
        ell.system(outline_sys_prompt),
        ell.user(outline_user_prompt)
    ]
    
@ell.simple(model=model_name, client=OpenAI(api_key=openai_api_key), temperature=0.3)
def evaluate_episode_outline(outlines : List[str]):
    eval_outline_user_prompt = f"""Evaluate each of these outlines for a 'Neural Kaleidoscope' episode: {'\nOUTLINE: \n'.join(outlines)}
    Score each outline on a scale of 1-10 for the following criteria:
        Thought-provoking nature
        Practical value for listeners
        Scientific grounding
        Originality of perspective
        Potential for listener engagement
    Select the best outline based on your evaluation.
    Output Format:
        BEST OUTLINE: 
        [Provide an exact, unmodified copy of the best outline]
        SCORES:
        Thought-provoking nature: [Score]
        Practical value for listeners: [Score]
        Scientific grounding: [Score]
        Originality of perspective: [Score]
        Potential for listener engagement: [Score]
        TOTAL SCORE: [Sum of all scores]
        OTHER OUTLINES:
        [One-line reason for not selecting each other outline]"""
    return [
        ell.system(eval_outline_sys_prompt),
        ell.user(eval_outline_user_prompt)
    ]
    
@ell.simple(model=model_name, client=OpenAI(api_key=openai_api_key), temperature=1.0)
def generate_episode_script_draft(best_outline : str):
    script_user_prompt = f"""Write a full podcast script draft for the 'Neural Kaleidoscope' episode based on this outline:
    {best_outline}
    Guidelines:
    Present the content in an article-style format, without host names or dialogue indicators.
    Use a sort of conversational tone that engages the listener directly.
    Incorporate metaphors or analogies to explain complex concepts, but ensure they are based on real-world phenomena or scientific principles.
    Provide actionable tips for listeners.
    Draw parallels between technology, human behavior, and personal growth.
    If using examples or case studies, only refer to well-known figures or documented research. Do not create fictional characters or scenarios.
    Aim for a 15-20 minute episode length (approximately 2000-2500 words).
    Do not include any audio cues, music breaks, or sound effect notes.
    Cite sources for any specific claims or statistics mentioned.
    Structure the script to flow naturally through the points in the outline, expanding on each with depth and insight.
    Output Format:
    [Title]
    [Full article-style script without any audio cues or host names]
    [List of sources cited]"""
    return [
        ell.system(script_sys_prompt),
        ell.user(script_user_prompt)
    ]

@ell.simple(model=model_name, client=OpenAI(api_key=openai_api_key), temperature=0.2)
def evaluate_episode_script_draft(script_drafts : List[str]):
    eval_script_user_prompt = f"""Thoroughly review these script drafts for the 'Neural Kaleidoscope' episode:
        {'\n=== SCRIPT DRAFT ===\n'.join(script_drafts)}
        Output Format:
        1. BEST SCRIPT
        [Provide an exact, unmodified copy of the best script]
        2. EVALUATION REPORT should not be more than 6 concise sentences.
        A. Review Notes
        
        B. Recommended Improvements
        Content Enhancements
        Specific suggestions for deepening insights
        Ideas for more engaging examples or metaphors
        Potential areas to expand upon
        Structure Refinements
        Flow improvements
        Pacing adjustments
        Transition enhancements

        C. Strong Elements from Other Scripts
        List compelling points, metaphors, or explanations from other drafts that could be incorporated
        Explain how these elements could be integrated into the best script

        D. Content Optimization
        Identify any low-value sections that could be removed or condensed
        Suggest replacements for removed content if necessary

        Guidelines:
        Maintain the original voice and style of the best script
        Focus on actionable, specific feedback
        Consider the target audience and episode goals when making suggestions
        Ensure all recommendations align with the 'Neural Kaleidoscope' podcast style"""
    return [
        ell.system(eval_script_sys_prompt),
        ell.user(eval_script_user_prompt)
    ]
    
@ell.simple(model=model_name, client=OpenAI(api_key=openai_api_key), temperature=1.0)
def generate_episode_description(episode : str):
    desc_user_prompt = f"Write an enticing episode description for 'Neural Kaleidoscope' based on this script: {episode}. The description should be attention-grabbing, hint at the value listeners will gain, and include emojis for visual appeal. Limit the description to 3-4 sentences."
    return [
        ell.system(desc_sys_prompt),
        ell.user(desc_user_prompt)
    ]
    
@ell.simple(model=model_name, client=OpenAI(api_key=openai_api_key), temperature=1.0)
def generate_episode_cover_art(concept: str):
    art_user_prompt = f"Create a detailed and creative image prompt for an AI image generator to produce cover art for this 'Neural Kaleidoscope' episode. Concept: {concept}. The image should be visually striking, relevant to the episode's theme, and suitable for a podcast episode cover. Include specific details about style, colors, elements, and composition. Only in 150 words"
    return [
        ell.system(art_sys_prompt),
        ell.user(art_user_prompt)
    ]
    
@ell.simple(model=model_name, client=OpenAI(api_key=openai_api_key), temperature=1.0)
def generate_episode_theme_song(concept: str):
    song_user_prompt = f"Create two distinct music prompts for Suno.ai for the 'Neural Kaleidoscope' episode based on the following concept: {concept} 1) An opening credit theme that sets the mood. 2) An ending credit theme that provides closure, bit of positivity, sort of enlightenment . Include specific musical elements like instruments, tempo, mood, and style. Avoid abstract descriptions; focus on concrete musical information.Only in 100 words"
    return [
        ell.system(song_sys_prompt),
        ell.user(song_user_prompt)
    ]


@ell.simple(model=model_name, client=OpenAI(api_key=openai_api_key), temperature=0.5)
def generate_episode(topic : str):
    
    concept = generate_episode_concept(topic)
    outlines = generate_episode_outline(concept, api_params=(dict(n=n_of_outlines)))
    best_outline = evaluate_episode_outline(outlines)
    script_drafts = generate_episode_script_draft(best_outline, api_params=(dict(n=n_of_script_drafts)))
    script_draft_review = evaluate_episode_script_draft(script_drafts)
    cover_art_prompt = generate_episode_cover_art(concept)
    theme_song_prompt= generate_episode_theme_song(concept)

    episode_user_prompt = f"Compile the final article for the 'Neural Kaleidoscope' episode based on this best script and the editor's review: {script_draft_review}. Incorporate the suggested improvements and ensure the script flows seamlessly in entertaining & engaging manner."

    # Saving thought flows of each role
    save_strings_to_files(directory=directory, concept=concept, outlines=outlines, best_outline= best_outline, script_drafts= script_drafts, script_draft_review= script_draft_review, cover_art_prompt= cover_art_prompt, theme_song_prompt= theme_song_prompt)

    return [
        ell.system(episode_sys_prompt),
        ell.user(episode_user_prompt)
    ]
 
episode = generate_episode(selected_topic)
save_strings_to_files(directory=directory, episode=episode)

print("Episode script created!")


"""
Role 1: Episode Concept Generator
Input: General topic or theme
Output: Specific episode concept

Role 2: Episode Outline Creator
Input: Episode concept
Output: A detailed episode outline

Role 3: Outline Evaluator and Selector
Input: List of episode outlines
Output: Best outline with evaluation report

Role 4: Script Writer
Input: Best episode outline
Output: List of full episode scripts

Role 5: Script Editor and Selector
Input: List of full episode scripts
Output: Best script with improvements

Role 6: Episode Description Writer
Input: Best full episode script
Output: Catchy episode description

Role 7: Cover Art Prompt Creator
Input: Episode concept and description
Output: Detailed image prompt for AI image generator

Role 8: Music Prompt Creator for Suno.ai
Input: Episode concept and description
Output: Opening and ending credit music prompts
"""