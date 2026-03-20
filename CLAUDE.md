# Project Guidelines

## Code Style

- Use object-oriented programming principles throughout: encapsulation, inheritance, polymorphism, and abstraction where appropriate.
- Never use shorthand variable names. Every variable, parameter, and attribute must have a full, descriptive name that communicates its purpose clearly.
- Write code that feels human and readable — prioritize clarity over cleverness. Code should read like prose, not a puzzle.

## Platform Compatibility

- All file paths, shell commands, and scripts must be written to work on both Windows and Unix-based systems.
- Avoid platform-specific syntax. Use cross-platform libraries and conventions wherever possible.

## Attribution and Credits

- Do not credit yourself, any tool, or any AI assistant anywhere in the code, comments, commit messages, or documentation.
- No co-authored-by lines, no generated-with notices, no tool signatures of any kind.

## Dependencies

- Keep `requirements.txt` up to date. Any time a new package is introduced or an existing one is removed, update the file to reflect that change.

## Git and GitHub

- Commit messages must be concise and describe what changed and why — nothing else.
- Pull request titles and descriptions must be short and focused on the change itself.
- Do not mention contributors, tools, or external systems in any GitHub-facing text unless strictly necessary for understanding the change.
