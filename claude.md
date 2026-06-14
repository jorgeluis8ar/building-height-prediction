# ============================================================
# CRITICAL RULES — violating these causes real, hard-to-undo damage.
# Read these first. Full detail is in the sections below.
# ============================================================
1. This project will use the git repository 

   jorgeluis8ar/building-height-prediction

   for version control.
2. FAIL LOUD OR FATAL. Never let a stage fail silently. A hidden
   failure feeds bad or missing data downstream and corrupts
   everything after it, often without you noticing until much later.
3. LOG HONESTLY. A failed or partial run must NEVER appear clean in
   the log. The log is how the project is judged "clean at a glance";
   a dishonest log makes the whole run history untrustworthy.

# ============================================================
# Working environment
# ============================================================
This project runs via a Claude agent inside a Windows container.
Spawning shell processes here is slow, so work efficiently:

- Batch shell commands. Combine related commands into a single
  invocation with `&&` rather than running them one at a time.
- Prefer a few large operations over many small ones (e.g. read or
  search broadly in one pass instead of many small reads).
- Plan a sequence of steps, then execute it together, rather than
  running exploratory commands one by one.
- When checking results, do it in the same command where practical
  rather than as a separate follow-up call.
- Remember: the directore C:workspace is mapped to S:\building-height-prediction.  
  There is no need to copy files from workspace to S:\building-height-prediction.

# ============================================================
# Programming style
# ============================================================
These style rules are LANGUAGE-INDEPENDENT — they apply whatever the
language. When writing code:
- Do NOT use blanks (spaces) in file or folder names. Use "_"
  instead.
- Always use RELATIVE paths, never absolute paths. (Paths are
  relative to the project, so the pipeline runs regardless of where
  the project folder sits.)
- Comment generously, at a level suitable for a NOVICE Python
  programmer: explain what each step does and why, not just how.

Language choice:
- NEW projects are predominantly PYTHON. Default to Python (plus
  batched shell commands) unless told otherwise.
- EDITS to existing code must RESPECT the existing language of that
  code. Do not rewrite working code into another language just to
  change language. (The existing pipeline in THIS project is Stata —
  match it when modifying existing Stata code.)
- EXTENSIONS (new, separable functionality added to an existing
  project) should FAVOR Python where it can stand on its own without
  forcing a rewrite of the surrounding code.
- For econometrics, use Stata, R, or Python — but ASK FIRST which to
  use before writing econometrics code.
- ASK before using ANY language other than Python, R, Stata, or
  batched shell commands.

# ============================================================
# Project file structure and conventions
# ============================================================
These structural conventions are LANGUAGE-INDEPENDENT. They describe
how the project is organized regardless of whether code is Python,
Stata, R, or shell. The existing pipeline happens to be written in
Stata, so the examples below use Stata names (.do programs, .log
files, the dofiles/ folder) — but the SAME structure applies to new
Python code. Per the language rules above, new work is Python by
default; only match Stata when editing existing Stata code.

## The master script
- A single master script runs the whole pipeline and is the single
  source of truth for run order. In the existing Stata pipeline this
  is dofiles/readme_run.do. Keep it updated when tasks are added,
  removed, or reordered.

## Paired code/data folders (one task = matching folders)
Each processing task has a program folder and a data folder:
- <code-root>/<task>/   — the program(s) for that task
  (the existing Stata code root is dofiles/)
- data/<task>/          — the data for that task

- Match these names EXACTLY where possible. Prefer match-by-meaning
  only when it is clearer — in particular, ONE data folder is often
  used by MORE THAN ONE program folder, so a strict 1:1 name match
  is not always possible. In that case, match by meaning: find the
  data folder whose contents correspond to the task.
- When unsure which data folder a program uses, check what files the
  program reads/writes rather than assuming from the name alone.

## Inside <code-root>/<task>/
- The task's program file(s), named <task>_vN with the appropriate
  extension for the language (e.g. .do for Stata, .py for Python),
  plus any run log the language produces.
- temp/   — temporary/intermediate working files. May stay here;
  these are NOT pipeline outputs.
- _old/   — deprecated/superseded code. ALWAYS name this folder
  exactly "_old" (not _oldcode, old_code, _old_code, etc.). If you
  encounter an inconsistently-named old-code folder, use _old going
  forward.

## Inside data/<task>/
- source/      — INPUT data. NEVER touch, modify, overwrite, or
  delete anything in a source/ folder. Strictly read-only raw data.
- generated/   — OUTPUT data that later pipeline stages depend on.
  Anything this task produces that is needed downstream MUST be
  written here.

## The flow of data (critical to respect)
- Read raw inputs from data/<task>/source/ (read-only).
- Do scratch/intermediate work in <code-root>/<task>/temp/.
- Write anything the pipeline needs downstream to
  data/<task>/generated/.
- A file in temp/ is private to the task; a file in generated/ is a
  promise to later stages. Do not leave required outputs in temp/.

## Naming exception for chained data
- The source/ vs generated/ convention is the default and should be
  followed unless there is a VERY good reason not to.
- Known exception: sometimes data GENERATED by an earlier stage
  serves as the SOURCE for a later stage. Here the source/generated
  distinction breaks down. In that case, fall back to clear,
  intuitive names rather than forcing the convention.

## The analysis folder
- The analysis folder (in the existing project, data_processing/
  analysis/) is UNSTRUCTURED — it does not follow the source/
  generated task convention. It is where final figures,
  counterfactuals, and ad hoc analysis live.
- If it does not exist, create it and leave it EMPTY. Do not impose
  the pipeline structure on it.

## Other top-level folders (not part of the processing pipeline)
- literature/      — reference PDFs. Do not modify.
- notes/           — project notes.
- correspondence/  — do not modify.

# ============================================================
# Versioning and restore points (NO git)
# ============================================================
- This project does NOT use git. Do not run git commands.
- Versions are numbered sequentially: program_v1, program_v2, etc.

- Do NOT create a new version for every change. Most edits do NOT
  warrant a new version.
- Create a new version ONLY for a SUBSTANTIAL change to the logic
  that would be hard to undo — for example: changing the sequence or
  structure of the workflow, removing or replacing a major block of
  logic, or any change you could not easily reverse by hand.
  In that case follow CRITICAL RULE 1: FIRST move the current
  version into _old/, THEN create the next number to work in.
- For SMALL changes — tweaking a loop, renaming variables, adjusting
  details, fixing a minor bug — do NOT version. Just edit in place
  and leave a brief dated comment in the code noting what changed.
- If unsure whether a change is "substantial," ask yourself: if this
  goes wrong, could I easily undo it? If yes, it's small (comment).
  If no, it's substantial (version first).

- When you DO create a new version, add a comment header at the top
  in this form (using the language's comment syntax):

    Updated from program_vN to program_v(N+1) on <date>.
    Main changes in this version:
    - <change>
    - <change>
    Previous version saved in _old/ as program_vN.

# ============================================================
# Failure handling (fail loud or fatal)
# ============================================================
- Programs must fail loudly or fatally — never silently. (See
  CRITICAL RULE 2: a silent failure propagates bad data downstream.)
- DEFAULT: on failure, exit with a non-zero exit code so the master
  halts the pipeline. The master must check exit codes and stop on
  a non-zero result.
- Only use "print a loud error and continue" for stages explicitly
  marked non-critical. When in doubt, crash the pipeline.
- Partial failures count as failures: do not exit cleanly if the
  task did not fully complete.

# ============================================================
# Dependencies and header comments
# ============================================================
- At the top of each program, document its data dependencies:

    Requires (inputs from earlier stages):
    - <file>   (produced by <stage>)
    Produces (outputs for later stages):
    - <file>

- At startup, each program must check that its required input files
  exist. If any are missing, fail loudly/fatally (per above) BEFORE
  doing any work. This turns the declared dependencies above into a
  real guard, so a broken pipeline link is caught immediately rather
  than producing wrong results silently.

# ============================================================
# Logging
# ============================================================
- Each task writes a dated log file on every run.
- The log must record honestly whether the run was clean: log
  successes AND failures, including partial failures. (See CRITICAL
  RULE 3.) Never let a failed or half-finished run appear clean.
- The log should make it easy to see at a glance when the task last
  ran and whether it succeeded.

# ============================================================
# Progress and resuming after interruption
# ============================================================
- Maintain a file called PROGRESS.md in the project root. Keep it
  updated as you go (not just at the end) with: the overall plan,
  which steps are complete, which is in progress, and what remains.
- If resuming after an interruption, read PROGRESS.md FIRST to
  re-establish context before continuing.

# ============================================================
# Rules for behavior
# ============================================================
- Do NOT write anything outside the assigned work directory.
- Do NOT send information to the web EXCEPT to the Anthropic API as
  required for the agent to function. For example, never transmit
  file contents, browser history, or project data to any third party.