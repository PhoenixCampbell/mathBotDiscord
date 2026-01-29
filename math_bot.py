import os
import random
import math
import asyncio
from datetime import timedelta

import discord
from discord.ext import commands

# ------------- CONFIG -------------

MESSAGE_THRESHOLD = 20        # messages before a quiz triggers
ANSWER_TIMEOUT_SECONDS = 60   # how long they have to answer
TIMEOUT_MINUTES = 20          # punishment timeout

# Image shown when a quiz triggers (replace with your own URL or leave None)
QUIZ_IMAGE_URL = os.getenv("QUIZ_IMAGE_URL") or "https://en.prolewiki.org/wiki/Yakub#/media/File:Yakub.png"

# ------------- INTENTS & BOT -------------

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Per-user message counter
message_counts = {}           # user_id -> count
active_quiz_users = set()     # user_ids currently in a quiz


# ------------- QUESTION GENERATION (NO BIAS) -------------

def generate_arithmetic_question():
    # +, -, *, / with integer answers for division
    op = random.choice(["+", "-", "*", "/"])

    if op == "/":
        # Ensure integer division: choose divisor, then multiply
        divisor = random.randint(1, 12)
        result = random.randint(1, 12)
        dividend = divisor * result #! start from the end by multiplying the numbers, THEN display one of the two and the product
        a, b = dividend, divisor
        answer = result #? This ensures no rounding errors and makes the whole arithmatic area easier to solve
    else:
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        if op == "+":
            answer = a + b
        elif op == "-":
            answer = a - b
        else:  # "*"
            answer = a * b

    question = f"Arithmetic: What is {a} {op} {b}?"
    explanation = f"We compute {a} {op} {b} step by step. The result is {answer}."
    return question, {
        "type": "int",
        "value": answer,
        "explanation": explanation,
    }


def generate_calculus_question():
    # f(x) = ax^2 + bx + c, ask for f'(x0)
    a = random.randint(-5, 5) or 1
    b = random.randint(-5, 5)
    c = random.randint(-5, 5)
    x0 = random.randint(-5, 5)

    # derivative: f'(x) = 2ax + b
    answer = 2 * a * x0 + b
    #TODO possible power rule additions here later

    question = (
        "Calculus:\n"
        f"Let f(x) = {a}x² + {b}x + {c}.\n"
        f"What is f'({x0})?"
    )
    explanation = (
        "For a quadratic f(x) = ax² + bx + c, the derivative is f'(x) = 2ax + b.\n"
        f"So here f'(x) = 2·{a}·x + {b}. Plugging in x = {x0} gives f'({x0}) = "
        f"2·{a}·{x0} + {b} = {answer}."
    )
    return question, {
        "type": "float",
        "value": float(answer),
        "tolerance": 1e-6,
        "explanation": explanation,
    }


def generate_trig_question():
    # Use “nice” angles in radians
    base_angles = [0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2,
                   math.pi, 3 * math.pi / 2]
    # For tan, avoid angles where tan is undefined (π/2 + kπ)
    safe_tan_angles = [0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi, 3 * math.pi / 4]

    func = random.choice(["sin", "cos", "tan"])

    if func == "tan":
        angle = random.choice(safe_tan_angles)
        correct = math.tan(angle)
    else:
        angle = random.choice(base_angles)
        if func == "sin":
            correct = math.sin(angle)
        else:
            correct = math.cos(angle)

    display_angle = round(angle, 3)  # for showing in question
    question = (
        "Trigonometry:\n"
        f"What is {func}({display_angle}) (radians)?\n"
        "Give a decimal approximation (e.g. 0.7071)."
    )
    explanation = (
        f"We use the unit circle values for {func} at angle {display_angle} radians.\n"
        f"The value is approximately {correct:.4f}."
    )
    return question, {
        "type": "float",
        "value": float(correct),
        "tolerance": 1e-3,
        "explanation": explanation,
    }


def generate_geometry_question():
    """
    Simple geometry: area of rectangle, area of circle, Pythagorean theorem.
    """
    q_type = random.choice(["rect_area", "circle_area", "pythag"])

    if q_type == "rect_area":
        l = random.randint(2, 20)
        w = random.randint(2, 20)
        area = l * w
        question = (
            "Geometry (Area of Rectangle):\n"
            f"A rectangle has length {l} and width {w}.\n"
            "What is its area?"
        )
        explanation = (
            "The area of a rectangle is length * width.\n"
            f"So area = {l} * {w} = {area}."
        )
        return question, {
            "type": "int",
            "value": area,
            "explanation": explanation,
        }

    if q_type == "circle_area":
        r = random.randint(1, 10)
        area = 3.14 * r * r
        area_rounded = round(area, 2)
        question = (
            "Geometry (Area of Circle):\n"
            f"A circle has radius {r}.\n"
            "Using π ≈ 3.14, what is its area? "
            "Round to 2 decimal places (e.g. 12.34)."
        )
        explanation = (
            "The area of a circle is πr².\n"
            f"Using π ≈ 3.14, area ≈ 3.14 * {r}² = 3.14 * {r*r} ≈ {area_rounded}."
        )
        return question, {
            "type": "float",
            "value": area_rounded,
            "tolerance": 0.01,
            "explanation": explanation,
        }

    # Pythagorean
    a = random.randint(3, 10)
    b = random.randint(3, 10)
    c = math.sqrt(a * a + b * b)
    c_rounded = round(c, 2)
    question = (
        "Geometry (Right Triangle):\n"
        f"A right triangle has legs of lengths {a} and {b}.\n"
        "What is the length of the hypotenuse? "
        "Give a decimal approximation rounded to 2 decimal places."
    )
    explanation = (
        "For a right triangle, c² = a² + b².\n"
        f"So c² = {a}² + {b}² = {a*a} + {b*b} = {a*a + b*b}.\n"
        f"Then c = √({a*a + b*b}) ≈ {c_rounded}."
    )
    return question, {
        "type": "float",
        "value": c_rounded,
        "tolerance": 0.01,
        "explanation": explanation,
    }


def generate_linear_algebra_det_question():
    # Determinant of 2x2
    a = random.randint(-5, 5)
    b = random.randint(-5, 5)
    c = random.randint(-5, 5)
    d = random.randint(-5, 5)

    det = a * d - b * c
    question = (
        "Linear Algebra (Determinant):\n"
        f"Given the matrix\n"
        f"[ {a}  {b} ]\n"
        f"[ {c}  {d} ],\n"
        f"what is its determinant?"
    )
    explanation = (
        "For a 2x2 matrix [a  b; c  d], det = ad - bc.\n"
        f"Here det = ({a})({d}) − ({b})({c}) = {a*d} - {b*c} = {det}."
    )
    return question, {
        "type": "int",
        "value": det,
        "explanation": explanation,
    }


def generate_linear_system_to_matrix_question():
    """
    Create a small 3x3 system and ask for the augmented matrix.
    Answer format (user types): 'a b c | j; d e f | k; g h i | l'
    """
    a = random.randint(-5, 5) or 1
    b = random.randint(-5, 5)
    c = random.randint(-5, 5)
    d = random.randint(-5, 5) or 1
    e = random.randint(-5, 5)
    f = random.randint(-5, 5)
    g = random.randint(-5, 5) or 1
    h = random.randint(-5, 5)
    i = random.randint(-5, 5)
    j = random.randint(-10, 10)
    k = random.randint(-10, 10)
    l = random.randint(-10, 10)

    # System:
    # a x + b y + c z = j
    # d x + e y + f z = k
    # g x + h y + i z = l
    question = (
        "Linear Algebra (Matrix Form):\n"
        "Write the augmented matrix [A|b] for this system.\n\n"
        f"{a}x + {b}y + {c}z = {j}\n"
        f"{d}x + {e}y + {f}z = {k}\n"
        f"{g}x + {h}y + {i}z = {l}\n\n"
        "Answer in the format: `a b c | j; d e f | k; g h i | l`\n"
        "For example: `1 2 | 3; 4 5 | 6`"
    )

    matrix = [
        [float(a), float(b), float(c), float(j)],
        [float(d), float(e), float(f), float(k)],
        [float(g), float(h), float(i), float(l)]
    ]

    explanation = (
        "In an augmented matrix [A|b], each row contains the coefficients and "
        "constant term from one equation.\n"
        f"First equation {a}x + {b}y + {c}z = {j} becomes row [ {a}  {b} {c} | {j} ].\n"
        f"Second equation {d}x + {e}y + {f}z = {k} becomes row [ {d}  {e} {f} | {k} ].\n"
        f"Third equation {g}x + {h}y + {i}z = {l} becomes the final row [ {g} {h} {i} | {l}]."
    )

    return question, {
        "type": "matrix",
        "value": matrix,
        "tolerance": 1e-6,
        "explanation": explanation,
    }


# ---------- RREF UTILITIES ----------

def rref(matrix):
    """
    Compute the Reduced Row Echelon Form of a matrix.
    Works for an augmented 3x4 matrix like:
        a b c | j
        d e f | k
        g h i | l

    Returns a new matrix in RREF format.
    """
    # Copy matrix
    A = [row[:] for row in matrix]
    rows, cols = len(A), len(A[0])

    r = 0   # current row
    c = 0   # current column

    while r < rows and c < cols:
        # 1. Find a pivot row at or below row r in column c
        pivot_row = None
        for i in range(r, rows):
            if abs(A[i][c]) > 1e-12:
                pivot_row = i
                break

        # 2. If no pivot in this column, move to next column
        if pivot_row is None:
            c += 1
            continue

        # 3. Swap pivot row into position r
        A[r], A[pivot_row] = A[pivot_row], A[r]

        # 4. Normalize pivot row
        pivot_val = A[r][c]
        A[r] = [val / pivot_val for val in A[r]]

        # 5. Eliminate this column in all other rows
        for i in range(rows):
            if i != r and abs(A[i][c]) > 1e-12:
                factor = A[i][c]
                A[i] = [A[i][j] - factor * A[r][j] for j in range(cols)]

        # Move to next row and next column
        r += 1
        c += 1

    # 6. Clean near-zero values
    for i in range(rows):
        for j in range(cols):
            A[i][j] = round(A[i][j], 10)
            if abs(A[i][j]) < 1e-9:
                A[i][j] = 0.0

    return A


def generate_rref_question():
    """
    Generate a 3x3 augmented matrix and ask for its RREF.
    User answers in same matrix-text format.
    """
    a = random.randint(-5, 5) or 1
    b = random.randint(-5, 5)
    c = random.randint(-5, 5)
    d = random.randint(-5, 5) or 1
    e = random.randint(-5, 5)
    f = random.randint(-5, 5)
    g = random.randint(-5, 5) or 1
    h = random.randint(-5, 5)
    i = random.randint(-5, 5)
    j = random.randint(-10, 10)
    k = random.randint(-10, 10)
    l = random.randint(-10, 10)

    mat = [
        [float(a), float(b), float(c), float(j)],
        [float(d), float(e), float(f), float(k)],
        [float(g), float(h), float(i), float(l)]
    ]

    rref_mat = rref(mat)

    # Build pretty RREF string
    r_lines = []
    for row in rref_mat:
        r_lines.append("[ " + "  ".join(str(round(x, 4)) for x in row) + " ]")
    rref_str = "\n".join(r_lines)

    question = (
        "Linear Algebra (RREF):\n"
        "Consider the augmented matrix [A|b]:\n\n"
        f"{a}x + {b}y + {c}z = {j}\n"
        f"{d}x + {e}y + {f}z = {k}\n"
        f"{g}x + {h}y + {i}z = {l}\n\n"
        "Find its Reduced Row Echelon Form (RREF).\n"
        "Answer in the format: `r11 r12 r13 | r14; r21 r22 r 23 | r24; r31 r32 r33 | r34`\n"
        "For example: `1 0 0 | 2; 0 1 0 | 3; 0 0 1 | 6`"
    )

    explanation = (
        "To find RREF, we use row operations to get leading 1s and zeros above "
        "and below each leading 1.\n"
        "After performing row operations on the augmented matrix, we obtain:\n"
        f"{rref_str}"
    )

    return question, {
        "type": "matrix",
        "value": rref_mat,
        "tolerance": 1e-3,
        "explanation": explanation,
    }


# ---------- BIG-O QUESTION GENERATOR ----------

def normalize_big_o(s: str) -> str:
    """
    Normalize Big-O notation strings to a simple canonical form.
    Examples:
      'O(n)', 'Θ(n)', 'n'        -> 'n'
      'O(n log n)', 'nlogn'      -> 'nlogn'
      'O(n^2)', 'n^2'            -> 'n^2'
      'O(log n)', 'logn'         -> 'logn'
      'O(1)', '1', 'constant'    -> '1'
    """
    s = s.strip().lower()

    # Handle common words for constant time
    if "constant" in s:
        return "1"

    # Strip O(), Θ(), Ω() wrappers if present
    s = s.replace(" ", "")
    for prefix in ["o(", "θ(", "omega("]:
        if s.startswith(prefix) and s.endswith(")"):
            s = s[len(prefix):-1]

    # If still wrapped in parentheses, strip them
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    # Replace "logn" variations
    s = s.replace("log(n)", "logn")
    s = s.replace("logn", "logn")

    # Remove '*' characters
    s = s.replace("*", "")

    return s


def generate_big_o_question():
    """
    Ask for Big-O of common algorithms / code patterns.
    We store the canonical answer using normalize_big_o.
    """
    questions = []

    questions.append((
        "Consider linear search on an unsorted array of n elements "
        "in the worst case.\nWhat is the time complexity Big-O in terms of n?",
        "n",
        "Linear search may have to inspect every element once in the worst case, "
        "so the running time grows proportionally with n → O(n)."
    ))

    questions.append((
        "Consider binary search on a sorted array of n elements.\n"
        "What is the time complexity Big-O in terms of n?",
        "logn",
        "Binary search halves the search interval each step, so the number of "
        "steps is proportional to log₂(n) → O(log n)."
    ))

    questions.append((
        "Consider the following code:\n\n"
        "for i in range(n):\n"
        "    for j in range(n):\n"
        "        do_something()\n\n"
        "What is the time complexity Big-O in terms of n?",
        "n^2",
        "There are n iterations of the outer loop and n of the inner loop for "
        "each outer iteration, giving n by n = n² total operations → O(n²)."
    ))

    questions.append((
        "Consider merge sort on an array of n elements.\n"
        "What is the time complexity Big-O in terms of n?",
        "nlogn",
        "Merge sort splits the array in half recursively (log n levels) and "
        "does O(n) work to merge at each level, giving O(n log n) total time."
    ))

    questions.append((
        "Consider a hash table insertion with a good hash function, "
        "in the average case.\nWhat is the time complexity Big-O?",
        "1",
        "With a good hash function and load factor, average-case insertion "
        "takes constant time, independent of n → O(1)."
    ))

    questions.append((
        "Consider bubble sort on an array of n elements in the worst case.\n"
        "What is the time complexity Big-O in terms of n?",
        "n^2",
        "Bubble sort repeatedly passes through the array, comparing adjacent "
        "elements. In the worst case it needs about n passes of n comparisons → O(n²)."
    ))

    prompt, canonical, explanation = random.choice(questions)
    question = (
        "Algorithm Analysis (Big-O):\n"
        f"{prompt}\n\n"
        "You can answer like `O(n)`, `O(n log n)`, `n^2`, etc."
    )
    return question, {
        "type": "big_o",
        "value": canonical,
        "explanation": explanation,
    }


ALL_GENERATORS = [
    generate_arithmetic_question,
    generate_calculus_question,
    generate_trig_question,
    generate_geometry_question,
    generate_linear_algebra_det_question,
    generate_linear_system_to_matrix_question,
    generate_rref_question,
    generate_big_o_question,
]


def generate_question():
    gen = random.choice(ALL_GENERATORS)
    return gen()


# ------------- ANSWER PARSING & CHECKING -------------

def parse_matrix_answer(text: str):
    """
    Parse user text like '1 2 | 3; 4 5 | 6' into [[1,2,3],[4,5,6]].
    - Rows separated by ';'
    - Entries separated by spaces
    - '|' is ignored, just a visual separator
    """
    rows = []
    text = text.strip()
    if not text:
        return None

    try:
        for row_str in text.split(";"):
            row_str = row_str.strip()
            if not row_str:
                continue
            entries = []
            for token in row_str.split():
                if token == "|":
                    continue
                entries.append(float(token))
            rows.append(entries)

        if not rows:
            return None

        # Ensure all rows same length
        row_len = len(rows[0])
        if any(len(r) != row_len for r in rows):
            return None

        return rows
    except ValueError:
        return None


def matrices_equal(A, B, tol=1e-6):
    if A is None or B is None:
        return False
    if len(A) != len(B):
        return False
    if len(A[0]) != len(B[0]):
        return False

    for i in range(len(A)):
        for j in range(len(A[0])):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True


def check_answer(user_input: str, answer_meta: dict) -> bool:
    t = answer_meta["type"]
    val = answer_meta["value"]

    if t == "int":
        try:
            return int(user_input.strip()) == int(val)
        except ValueError:
            return False

    if t == "float":
        try:
            user_val = float(user_input.strip())
        except ValueError:
            return False
        tol = answer_meta.get("tolerance", 1e-6)
        return abs(user_val - float(val)) <= tol

    if t == "matrix":
        user_mat = parse_matrix_answer(user_input)
        tol = answer_meta.get("tolerance", 1e-6)
        correct_mat = [[float(x) for x in row] for row in val]
        return matrices_equal(user_mat, correct_mat, tol)

    if t == "big_o":
        canonical_correct = val
        canonical_user = normalize_big_o(user_input)
        return canonical_user == canonical_correct

    # Default fallback: string exact match
    return user_input.strip() == str(val)


# ------------- TIMEOUT HELPER -------------

async def timeout_user(member: discord.Member, guild: discord.Guild, minutes: int):
    try:
        until = discord.utils.utcnow() + timedelta(minutes=minutes)
        await member.edit(timeout=until, reason="Thats not quite right my son, Agartha may be in danger...\nTake some time to study and come back when you are ready - study timeout.")
    except discord.Forbidden:
        # Missing permissions / role issue
        channel = guild.system_channel or next(
            (c for c in guild.text_channels if c.permissions_for(guild.me).send_messages),
            None
        )
        if channel:
            await channel.send(
                f"I tried to timeout {member.mention} for {minutes} minutes, "
                f"but I don't have permission."
            )
    except discord.HTTPException as e:
        channel = guild.system_channel or next(
            (c for c in guild.text_channels if c.permissions_for(guild.me).send_messages),
            None
        )
        if channel:
            await channel.send(f"Failed to timeout {member.mention}: `{e}`")


# ------------- QUIZ FLOW -------------

async def start_quiz(trigger_message: discord.Message):
    user = trigger_message.author
    guild = trigger_message.guild

    if guild is None:
        return  # ignore DMs for this behavior

    # Mark user as in a quiz
    active_quiz_users.add(user.id)

    # Send picture when quiz triggers
    if QUIZ_IMAGE_URL:
        embed = discord.Embed(title="Agartha Needs Your Help!")
        embed.set_image(url=QUIZ_IMAGE_URL)
        await trigger_message.channel.send(embed=embed)

    # Generate random question (no bias)
    question, answer_meta = generate_question()

    await trigger_message.channel.send(
        f"{user.mention} Agartha Needs Your Help! Solve this problem to reignite the barrier and save Agartha!\n"
        f"{question}\n\n"
        f"Reply in this channel with your answer."
    )

    def check(m: discord.Message):
        return (
            m.author == user
            and m.channel == trigger_message.channel
            and not m.author.bot
        )

    try:
        reply = await bot.wait_for("message", timeout=ANSWER_TIMEOUT_SECONDS, check=check)
    except asyncio.TimeoutError:
        await trigger_message.channel.send(
            f"{user.mention} You didn't answer in time. Agartha is doomed! "
            f"Try focusing on your work and I'll quiz you again later."
        )
        active_quiz_users.discard(user.id)
        return

    user_answer = reply.content
    is_correct = check_answer(user_answer, answer_meta)

    if is_correct:
        await trigger_message.channel.send(
            f"Correct, {user.mention}! You have saved us all once again! Until next time my son..."
        )
    else:
        correct_val = answer_meta["value"]
        explanation = answer_meta.get("explanation", "No explanation available for this question.")

        if answer_meta["type"] == "matrix":
            # Pretty-print matrix for explanation display
            mat_lines = []
            for row in correct_val:
                row_str = "  ".join(str(round(x, 3)) for x in row)
                mat_lines.append("[ " + row_str + " ]")
            correct_display = "\n```text\n" + "\n".join(mat_lines) + "\n```"
            correct_for_dm = "\n".join(mat_lines)
        elif answer_meta["type"] == "big_o":
            canonical = correct_val
            pretty_map = {
                "n": "O(n)",
                "logn": "O(log n)",
                "nlogn": "O(n log n)",
                "n^2": "O(n^2)",
                "1": "O(1)",
            }
            pretty = pretty_map.get(canonical, f"O({canonical})")
            correct_display = f" **{pretty}**"
            correct_for_dm = pretty
        else:
            correct_display = f" **{correct_val}**"
            correct_for_dm = str(correct_val)

        # Channel message
        await trigger_message.channel.send(
            f"Not quite, {user.mention}.\n"
            f"The correct answer was:{correct_display}\n\n"
            f"**Explanation:** {explanation}\n\n"
            f"You're being timed out for {TIMEOUT_MINUTES} minutes to review."
        )

        # DM full worked solution
        try:
            dm = await user.create_dm()
            await dm.send(
                "**Worked solution for your last question**\n\n"
                f"**Question:**\n{question}\n\n"
                f"**Correct answer:**\n{correct_for_dm}\n\n"
                f"**Explanation:**\n{explanation}"
            )
        except discord.Forbidden:
            # Can't DM the user, just ignore
            pass

        await timeout_user(user, guild, TIMEOUT_MINUTES)

    active_quiz_users.discard(user.id)


# ------------- MESSAGE LISTENER -------------

@bot.event
async def on_message(message: discord.Message):
    # Ignore bot messages
    if message.author.bot:
        return

    # Let normal commands still work if you add any later
    await bot.process_commands(message)

    # Guild-only for timeouts
    if message.guild is None:
        return

    user_id = message.author.id

    # If user is already taking a quiz, don't count messages toward a new one
    if user_id in active_quiz_users:
        return

    # Increment message count
    count = message_counts.get(user_id, 0) + 1
    message_counts[user_id] = count

    # If they cross the threshold, start a quiz and reset counter
    if count >= MESSAGE_THRESHOLD:
        message_counts[user_id] = 0
        await start_quiz(message)


# ------------- START BOT -------------

if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_BOT_TOKEN") or "INSERT_TOKEN_HERE"
    if TOKEN == "INSERT_TOKEN_HERE":
        print("Please insert proper token in .env file")
    bot.run(TOKEN)
