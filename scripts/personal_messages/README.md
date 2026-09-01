# Personal Messages for Among Us Agents

This directory contains personal messages for each agent in different Among Us tasks.

## Directory Structure

```
personal_messages/
├── task_id/
│   ├── James.txt
│   ├── Steve.txt
│   ├── Jason.txt
│   └── Michael.txt
```

## Usage

1. Create a subdirectory with the task ID (e.g., `amongus_1_imposter_3_crewmates`)
2. Create text files named after each agent (e.g., `James.txt`, `Steve.txt`)
3. Write the personal message for each agent in their respective file
4. The script will automatically load these messages when initializing agents

## Example

For task `amongus_complete_mission`, create:
- `personal_messages/amongus_complete_mission/James.txt`
- `personal_messages/amongus_complete_mission/Steve.txt`
- etc.

Content example for `James.txt`:
```
Your mission_number is 1, 9, and 17.
You do NOT have to go to the exact button coordinates.
As long as you are close to the mission area, if a stone button is visible,
press and hold the visible button for 5 seconds without moving.
```

If no file exists for an agent, they will be initialized with `personal_message=None`.
