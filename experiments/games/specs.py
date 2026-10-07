CLM_RULES = 'T-Rex runner: avoid collisions. Choose the next control for six frames. Run continues forward motion and any existing jump. Jump starts a jump from the ground. Duck crouches on the ground or accelerates descent in the air. Cacti are ground obstacles; birds may leave room underneath. The game pauses while the model decides. Obstacles move left. The dinosaur is 44 pixels wide and 47 pixels tall when standing. Feet height zero means grounded. Contact ends the game. A jump rises, reaches its peak, falls, and ends; it cannot be extended by repeatedly choosing jump. Duck does not stop horizontal motion. The examples below describe other situations. Choose for the current observation at the end, using the examples to understand timing.'
CLM_PROMPT_ADDITIONS = ['Example state: The dinosaur is on the ground. A cactus is 60 pixels beyond its front, 25 pixels wide and '
 '50 pixels tall, moving toward it at 6 pixels per frame. Answer: jump. Explanation: Start rising to clear '
 'the approaching cactus; staying on the ground would cause a collision.',
 'Example state: The dinosaur is on the ground. A cactus is 500 pixels beyond its front, moving toward it at '
 '6 pixels per frame. Answer: run. Explanation: A jump now would finish before the cactus arrived.',
 'Example state: The dinosaur is on the ground. A cactus is 180 pixels beyond its front, 25 pixels wide and '
 '50 pixels tall, approaching at 6 pixels per frame. Answer: run. Explanation: Wait for this six-frame turn. '
 'Jumping now would put the descent too early, while the cactus is still passing.',
 'Example state: The dinosaur is in the air with its feet 80 pixels above the ground, falling at 2 pixels '
 'per frame. A cactus is 20 pixels beyond its front, 25 pixels wide and 50 pixels tall, approaching at 6 '
 'pixels per frame. Answer: run. Explanation: Let the current jump continue over the cactus. Ducking would '
 'force the dinosaur down into it.',
 'Compare two cases with the same grounded dinosaur, cactus size, and speed of 6 pixels per frame. At a '
 'distance of 180 pixels, choose run: the cactus arrives too late for a jump started now. At a distance of '
 '60 pixels, choose jump: the cactus will arrive soon enough for this jump to clear it. The distance changes '
 'the answer even though both observations mention a cactus.',
 'Example state: The dinosaur is on the ground. A cactus is 40 pixels beyond its front, approaching at 6 '
 'pixels per frame. Answer: jump. Explanation: Waiting another six frames would leave almost no clearance; '
 'begin rising now.',
 'Example state: The dinosaur is on the ground. A cactus is 300 pixels beyond its front, approaching at 6 '
 'pixels per frame. Answer: run. Explanation: An obstacle being visible does not mean it is time to jump. It '
 'will still be far ahead after this six-frame turn.',
 'Example state: The dinosaur is on the ground. A cactus is 120 pixels beyond its front, 25 pixels wide and '
 '50 pixels tall, moving toward it at 6 pixels per frame. Answer: run. Explanation: Keep running for this '
 'six-frame turn. There is time to get closer before starting a jump, so it will not land too early.',
 'Example state: The dinosaur is on the ground. A cactus is 200 pixels beyond its front, 34 pixels wide and '
 '35 pixels tall, moving toward it at 6 pixels per frame. Answer: run. Explanation: This shorter cactus is '
 'still too far away. Continue running for six frames and look again before jumping.']
CLM_PROMPT_ADDITIONS.append('Example state: The dinosaur is on the ground. A cactus is 150 pixels beyond its front, 17 pixels wide and 35 pixels tall, moving toward it at 6 pixels per frame. Answer: run. Explanation: This small cactus is still too far away. A jump started now would finish while the cactus was passing the dinosaur. Wait for the next observation before starting the jump.')
CLM_ACTION_TEXTS = {'run': 'Run.', 'jump': 'Jump.', 'duck': 'Duck.'}
CLM_QUESTION = 'What should the dinosaur do for the next six frames only? It will get another observation afterward. Decide whether to keep running, jump now, or duck now.'
def clm_example(example):
    return clm_observation(example)

def clm_observation(observation):
    import re
    text=observation.removesuffix('\nAnswer:')
    match=re.search(r'is (-?\d+) pixels beyond its front',text)
    if match:
        gap=int(match.group(1))
        if gap>=0:text+='\nSide view of the gap, each dash is approximately 10 pixels: dinosaur |'+'-'*round(gap/10)+'| obstacle.'
        else:text+='\nThe obstacle front has passed the dinosaur front by '+str(-gap)+' pixels.'
    return text
TREX_ACTIONS = {
    'run': 'Release duck. Running is automatic; an existing jump continues.',
    'jump': 'Release duck and start a jump if grounded. No second jump while airborne.',
    'duck': 'Crouch if grounded; start a rapid descent if airborne.',
}
TREX_RULES = """You are playing T-Rex runner. Stay alive by avoiding collisions. At each turn, choose A, B, or C for the next 6 frames, then you will receive a new state and choose again. The game pauses while you decide.

A continues automatic running, or lets an existing jump continue.
B starts a jump from the ground. It does nothing extra in the air.
C crouches on the ground, or rapidly descends in the air.

Obstacles move left. The state gives distances, sizes, and heights above ground. The dinosaur is 44 pixels wide and 47 pixels tall when standing. Feet height zero means grounded. Contact ends the game. A jump rises, reaches its peak, falls, and ends; it cannot be extended by repeatedly selecting B. Duck does not stop horizontal motion.

Output only the chosen letter."""
TREX_EXAMPLES = [
    "State: The dinosaur is on the ground. A cactus is 500 pixels beyond its front, moving toward it at 6 pixels per frame.\nAnswer: A (run).\nExplanation: A jump now would finish before the cactus arrived. Continue running for this turn and reconsider at the next observation.",
    "State: The dinosaur is on the ground. A cactus is 60 pixels beyond its front, 25 pixels wide and 50 pixels tall, moving toward it at 6 pixels per frame.\nAnswer: B (jump).\nExplanation: Begin rising now so the dinosaur can pass above the approaching cactus. Continuing on the ground would bring it into the cactus.",
    "State: The dinosaur is on the ground. A cactus is 180 pixels beyond its front, 25 pixels wide and 50 pixels tall, approaching at 6 pixels per frame.\nAnswer: A (run).\nExplanation: This is still too early to start the jump. The dinosaur would descend while the cactus was passing. Keep running for this six-frame turn, then reassess the smaller distance.",
    "State: The dinosaur is in the air, with its feet 80 pixels above the ground, falling at 2 pixels per frame. A cactus is 20 pixels beyond its front, 25 pixels wide and 50 pixels tall, moving toward it at 6 pixels per frame.\nAnswer: A (run).\nExplanation: Let the current jump carry the dinosaur over the cactus. Ducking would accelerate its descent into the obstacle; another jump command cannot add height while airborne.",
]
MARIO_ACTIONS={'wait':'Release all buttons; remain still or coast with current momentum.', 'right':'Walk right with jump released.', 'left':'Walk left with jump released.', 'jump':'Hold jump without a horizontal direction.', 'jump_right':'Hold jump and right to jump forward or continue an airborne jump to the right.', 'jump_left':'Hold jump and left to jump backward or continue an airborne jump to the left.'}
MARIO_RULES='''You control Super Mario Bros. Reach the flag to the right without dying. Each decision holds its buttons for 12 frames (200 ms) at 60 frames per second. Observations use world pixel coordinates; x increases right, y increases downward. Mario is about 16 pixels wide. Feet_y is the bottom of his body. Grounded means standing on a solid surface. Solid terrain rectangles can be floor, pipes, bricks, or blocks. Missing floor is a pit: falling into it is fatal. Jump over enemies, pits, and tall obstacles. Enemies can be stomped by landing on them from above; side contact is dangerous. Walk right on clear floor; use jumps when needed for progress or safety. Holding jump while rising increases jump height; releasing jump shortens the jump. Jump must be released between separate jumps: continuing to hold it after landing does not start a new jump. Horizontal momentum persists briefly after release. Negative vertical velocity means rising. The state gives only currently visible geometry and current motion.'''
SPECS={'trex':{'rules':TREX_RULES,'actions':TREX_ACTIONS,'question':'Which action should the dinosaur take now to survive?'},'mario':{'rules':MARIO_RULES,'actions':MARIO_ACTIONS,'question':'Which action should Mario take now to reach the flag safely?'}}
