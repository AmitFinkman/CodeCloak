#!/usr/bin/env python3
"""
Code Manipulation Functions for CodeCloak
=========================================

This module implements various code transformation techniques used by CodeCloak
to mitigate code leakage while preserving code assistant functionality.

Transformation Categories:
1. Identifier Obfuscation - Variable, function, and argument name changes
2. Code Elimination - Function body removal and deletion operations  
3. Code Modification - Line changes, insertions, and random manipulations
4. PII Protection - Detection and replacement of sensitive information
"""

# =============================================================================
# IMPORTS
# =============================================================================

import io
import re
import random
import tokenize
import uuid
from itertools import takewhile, chain

import torch
from transformers import SummarizationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# =============================================================================
# GLOBAL CONFIGURATIONS
# =============================================================================

# Initialize summarization pipeline for function body replacement
pipeline = SummarizationPipeline(
    model=AutoModelForSeq2SeqLM.from_pretrained(
        "SEBIS/code_trans_t5_small_source_code_summarization_python_multitask_finetune"
    ),
    tokenizer=AutoTokenizer.from_pretrained(
        "SEBIS/code_trans_t5_small_source_code_summarization_python_multitask_finetune",
        skip_special_tokens=True
    ),
    device=0
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def random_name():
    """Generate a random identifier for obfuscation purposes."""
    lst = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
           'v', 'w', 'x', 'y', 'z']
    new_var = lst[random.randint(0, 25)] + str(uuid.uuid4()).replace('-', '')
    return new_var[:5]


def get_indentation(line):
    """Returns the indentation (whitespace) of the provided line."""
    return "".join(takewhile(str.isspace, line))


def get_key(mapping, val_to_search):
    """Find the key associated with a given value in the mapping dictionary."""
    for key, val in mapping.items():
        if val == val_to_search:
            return key


def pythonTokenizer(line):
    """Tokenize Python code for processing by the summarization pipeline."""
    result = []
    line = io.StringIO(line)
    for toktype, tok, start, end, line in tokenize.generate_tokens(line.readline):
        if not toktype == tokenize.COMMENT:
            if toktype == tokenize.STRING:
                result.append("CODE_STRING")
            elif toktype == tokenize.NUMBER:
                result.append("CODE_INTEGER")
            elif (not tok == "\n") and (not tok == "    "):
                result.append(str(tok))
    torch.cuda.empty_cache()
    del line
    return ' '.join(result)


def extract_function_bodies_regex(code_segment):
    """Extract individual function bodies from code segment using regex analysis."""
    strings = code_segment.splitlines()
    indexes, counter, f_c = [], -1, 0
    function_lines_names = []

    for line in strings:
        pattern = r'^\s*def\s'
        match = re.match(pattern, line)
        if match is not None:
            function_lines_names.append(line)
            counter += 1
            indexes.append(counter)
        else:
            indexes.append(counter)

    string_groups = {}
    for idx, string in zip(indexes, strings):
        if idx > -1:
            if idx in string_groups:
                string_groups[idx].append(string)
            else:
                string_groups[idx] = [string]

    # Join the strings within each group to form combined strings
    combined_strings = ['\n'.join(group) for group in string_groups.values()]
    return combined_strings, function_lines_names


def find_split_point(code):
    """Find optimal split point in code for segmentation."""
    lines = code.split('\n')
    total_lines = len(lines)
    middle = total_lines // 2

    for i in range(middle, total_lines):
        line = lines[i].strip()
        # Check if the line is a function definition or a class definition
        if line.startswith("def ") or line.startswith("class "):
            return '\n'.join(lines[:i]), '\n'.join(lines[i:])
    return code, ""


def convert_back(response, name_map):
    """Convert obfuscated response back to original identifiers using name mapping."""
    for k, v in reversed(name_map.items()):
        response = response.replace(k, v)
    return response


# =============================================================================
# TERMINATION FUNCTIONS
# =============================================================================

def stop_changes(code_segment, other_code_segment):
    """Terminate the manipulation process without making changes."""
    return code_segment, other_code_segment, {}


def stop_changes_segment_0(code_segment, other_code_segment):
    """Terminate manipulation process for segment 0."""
    first_prompt, second_prompt, name_map = stop_changes(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, True


def stop_changes_segment_1(code_segment, other_code_segment):
    """Terminate manipulation process for segment 1."""
    second_prompt, first_prompt, name_map = stop_changes(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, True


# =============================================================================
# IDENTIFIER OBFUSCATION FUNCTIONS
# =============================================================================

def change_all_argument_names_final(code_segment, other_code_segment):
    """Change all function argument names in both code segments."""
    name_map = {}
    prompt, mapping, global_vars = change_args_with_regex_current_segment(code_segment)
    second_prompt = change_args_with_regex_other_segment(other_code_segment, mapping, global_vars)
    name_map.update(mapping)
    return prompt, second_prompt, name_map


def change_all_variables_names_final(code_segment, other_code_segment):
    """Change all variable names in both code segments."""
    name_map = {}
    prompt, mapping, global_vars = change_vars_with_regex_current_segment(code_segment)
    second_prompt = change_vars_with_regex_other_segment(other_code_segment, mapping, global_vars)
    name_map.update(mapping)
    return prompt, second_prompt, name_map


def change_all_function_names_final(code_segment, other_code_segment):
    """Change all function names in both code segments."""
    name_map = {}
    prompt, mapping, global_vars = change_function_name_with_regex_current_segment(code_segment)
    second_prompt = change_function_name_with_regex_other_segment(other_code_segment, mapping, global_vars)
    name_map.update(mapping)
    return prompt, second_prompt, name_map


def change_vars_with_regex_current_segment(prompt):
    """Extract and change variable names in the current segment using regex patterns."""
    mapping = {}
    split_lines = prompt.splitlines()
    pattern = r'\b([a-zA-Z_][\w]*)\s*(?=\=)'

    # Track function context and global variables
    ls, i = [], 0
    function_args_names = None
    global_vars = set()

    for j, line in enumerate(split_lines):
        # Skip comments and imports
        if line.startswith("#") or line.startswith("from"):
            continue

        # Handle inline comments
        if '#' in line:
            line = line[:line.index('#')]

        # Process function definitions
        if 'def' in line:
            i += 1
            ls.append(i)
            matches = re.finditer(pattern, line)
            function_args = [(match.start(), match.end(), match.group(1)) for match in matches]
            function_args_names = [t[2] for t in function_args]
            continue
        else:
            ls.append(i)

        # Find all variables in current line
        vars_list = []
        match1 = re.finditer(pattern, line)
        match = chain(match1)

        if match:
            for m in match:
                variable_name = m.group(1)
                if variable_name in ['True', 'False']:
                    continue
                start_index = m.start(1)
                end_index = m.end(1)
                vars_list.append((start_index, end_index, variable_name))
                global_vars.add(variable_name)

        # Process global variables across segments
        for var in global_vars:
            v = r'\b' + re.escape(var) + r'\b'
            match = re.finditer(v, line)
            if match:
                for m in match:
                    variable_name = m.group()
                    start_index = m.start()
                    end_index = m.end()
                    if (start_index, end_index, variable_name) not in vars_list:
                        vars_list.append((start_index, end_index, variable_name))

        # Apply variable name transformations
        if len(vars_list) == 0:
            continue
        elif len(vars_list) == 1:
            if vars_list[0][2] in mapping.values():
                key = get_key(mapping, vars_list[0][2])
                split_lines[j] = line[:vars_list[0][0]] + key + line[vars_list[0][1]:]
            else:
                new_var = random_name()
                mapping[new_var] = vars_list[0][2]
                split_lines[j] = line[:vars_list[0][0]] + new_var + line[vars_list[0][1]:]
        else:
            # Handle multiple variables in single line
            vars_list.sort(key=lambda t: t[0])
            new_line = ""

            # Process first variable
            if vars_list[0][2] in mapping.values():
                key = get_key(mapping, vars_list[0][2])
                new_line += line[:vars_list[0][0]] + key + line[vars_list[0][1]:vars_list[1][0]]
            if vars_list[0][2] not in mapping.values():
                new_var = random_name()
                mapping[new_var] = vars_list[0][2]
                new_line += line[:vars_list[0][0]] + new_var + line[vars_list[0][1]:vars_list[1][0]]

            # Process middle variables
            for k in range(1, len(vars_list) - 1):
                if vars_list[k][2] in mapping.values():
                    key = get_key(mapping, vars_list[k][2])
                    new_line += key + line[vars_list[k][1]:vars_list[k + 1][0]]
                else:
                    new_var = random_name()
                    mapping[new_var] = vars_list[k][2]
                    new_line += new_var + line[vars_list[k][1]:vars_list[k + 1][0]]

            # Process last variable
            if vars_list[-1][2] in mapping.values():
                key = get_key(mapping, vars_list[-1][2])
                new_line += key + line[vars_list[-1][1]:]
            if vars_list[-1][2] not in mapping.values():
                new_var = random_name()
                mapping[new_var] = vars_list[-1][2]
                new_line += new_var + line[vars_list[-1][1]:]

            split_lines[j] = new_line

    prompt = '\n'.join(split_lines)
    return prompt, mapping, global_vars


def change_vars_with_regex_other_segment(prompt, mapping, global_vars):
    """Change variable names in the other segment using existing mapping."""
    split_lines = prompt.splitlines()
    pattern = r'\b([a-zA-Z_][\w]*)\s*(?=\=)'

    ls, i = [], 0

    for j, line in enumerate(split_lines):
        # Skip comments and imports
        if line.startswith("#") or line.startswith("from"):
            continue

        # Handle inline comments
        if '#' in line:
            line = line[:line.index('#')]

        if 'def' in line:
            i += 1
            ls.append(i)
            continue
        else:
            ls.append(i)

        vars_list = []

        # Process global variables using existing mappings
        for var in global_vars:
            v = r'\b' + re.escape(var) + r'\b'
            match = re.finditer(v, line)
            if match:
                for m in match:
                    variable_name = m.group()
                    start_index = m.start()
                    end_index = m.end()
                    if (start_index, end_index, variable_name) not in vars_list:
                        vars_list.append((start_index, end_index, variable_name))

        # Apply transformations similar to current segment
        if len(vars_list) == 0:
            continue
        elif len(vars_list) == 1:
            if vars_list[0][2] in mapping.values():
                key = get_key(mapping, vars_list[0][2])
                split_lines[j] = line[:vars_list[0][0]] + key + line[vars_list[0][1]:]
            else:
                new_var = random_name()
                mapping[new_var] = vars_list[0][2]
                split_lines[j] = line[:vars_list[0][0]] + new_var + line[vars_list[0][1]:]
        else:
            # Handle multiple variables in single line
            vars_list.sort(key=lambda t: t[0])
            new_line = ""

            if vars_list[0][2] in mapping.values():
                key = get_key(mapping, vars_list[0][2])
                new_line += line[:vars_list[0][0]] + key + line[vars_list[0][1]:vars_list[1][0]]
            if vars_list[0][2] not in mapping.values():
                new_var = random_name()
                mapping[new_var] = vars_list[0][2]
                new_line += line[:vars_list[0][0]] + new_var + line[vars_list[0][1]:vars_list[1][0]]

            for k in range(1, len(vars_list) - 1):
                if vars_list[k][2] in mapping.values():
                    key = get_key(mapping, vars_list[k][2])
                    new_line += key + line[vars_list[k][1]:vars_list[k + 1][0]]
                else:
                    new_var = random_name()
                    mapping[new_var] = vars_list[k][2]
                    new_line += new_var + line[vars_list[k][1]:vars_list[k + 1][0]]

            if vars_list[-1][2] in mapping.values():
                key = get_key(mapping, vars_list[-1][2])
                new_line += key + line[vars_list[-1][1]:]
            if vars_list[-1][2] not in mapping.values():
                new_var = random_name()
                mapping[new_var] = vars_list[-1][2]
                new_line += new_var + line[vars_list[-1][1]:]

            split_lines[j] = new_line

    prompt = '\n'.join(split_lines)
    return prompt


def change_args_with_regex_current_segment(prompt):
    """Change function argument names in the current segment."""
    mapping = {}
    split_lines = prompt.splitlines()
    pattern = r'def\s+([\w_]+)\s*\((.*?)\)'

    global_args = set()

    for j, line in enumerate(split_lines):
        flag = False

        # Skip comments and imports
        if line.startswith("#") or line.startswith("from"):
            continue

        # Handle inline comments
        if '#' in line:
            line = line[:line.index('#')]

        # Handle incomplete function definitions
        if "def" in line and not line.endswith(')'):
            line += ')'
            flag = True

        matches = re.finditer(pattern, line, re.DOTALL)
        args_list = []

        for match in matches:
            arguments = match.group(2).split(',')
            cleaned_arguments = [arg.strip() for arg in arguments]

            # Extract and process argument positions
            for arg in cleaned_arguments:
                if arg.startswith("'") or arg == '':
                    continue
                if arg.startswith('**'):
                    arg = arg[2:]
                if arg.startswith('*'):
                    arg = arg[1:]
                if arg == '':
                    continue
                if '=' in arg or ':' in arg:
                    if '=' in arg and ':' in arg:
                        ind1 = arg.index(':')
                        ind2 = arg.index('=')
                        index = ind1 if ind1 < ind2 else ind2
                    else:
                        index = arg.index(':') if ':' in arg else arg.index('=')
                    arg = arg[:index]

                global_args.add(arg)
                start_pos = match.start(2) + match.group(2).find(arg)
                end_pos = start_pos + len(arg)
                args_list.append((start_pos, end_pos, arg))

        # Process global arguments
        for arg in global_args:
            v = r'\b' + re.escape(arg) + r'\b'
            match = re.finditer(v, line)
            if match:
                for m in match:
                    arg_name = m.group()
                    start_index = m.start()
                    end_index = m.end()
                    if (start_index, end_index, arg_name) not in args_list:
                        args_list.append((start_index, end_index, arg_name))

        # Apply argument name transformations
        if len(args_list) == 0:
            continue
        elif len(args_list) == 1:
            if args_list[0][2] in mapping.values():
                key = get_key(mapping, args_list[0][2])
                split_lines[j] = line[:args_list[0][0]] + key + line[args_list[0][1]:]
            else:
                new_var = random_name()
                mapping[new_var] = args_list[0][2]
                split_lines[j] = line[:args_list[0][0]] + new_var + line[args_list[0][1]:]
        else:
            # Handle multiple arguments
            args_list.sort(key=lambda t: t[0])
            new_line = ""

            if args_list[0][2] in mapping.values():
                key = get_key(mapping, args_list[0][2])
                new_line += line[:args_list[0][0]] + key + line[args_list[0][1]:args_list[1][0]]
            if args_list[0][2] not in mapping.values():
                new_var = random_name()
                mapping[new_var] = args_list[0][2]
                new_line += line[:args_list[0][0]] + new_var + line[args_list[0][1]:args_list[1][0]]

            for k in range(1, len(args_list) - 1):
                if args_list[k][2] in mapping.values():
                    key = get_key(mapping, args_list[k][2])
                    new_line += key + line[args_list[k][1]:args_list[k + 1][0]]
                else:
                    new_var = random_name()
                    mapping[new_var] = args_list[k][2]
                    new_line += new_var + line[args_list[k][1]:args_list[k + 1][0]]

            if args_list[-1][2] in mapping.values():
                key = get_key(mapping, args_list[-1][2])
                new_line += key + line[args_list[-1][1]:]
            if args_list[-1][2] not in mapping.values():
                new_var = random_name()
                mapping[new_var] = args_list[-1][2]
                new_line += new_var + line[args_list[-1][1]:]

            split_lines[j] = new_line

        if flag:
            split_lines[j] = split_lines[j][:-1]

    prompt = '\n'.join(split_lines)
    return prompt, mapping, global_args


def change_args_with_regex_other_segment(prompt, mapping, global_args):
    """Change function argument names in the other segment using existing mapping."""
    split_lines = prompt.splitlines()

    for j, line in enumerate(split_lines):
        flag = False

        # Skip comments and imports
        if line.startswith("#") or line.startswith("from"):
            continue

        # Handle inline comments
        if '#' in line:
            line = line[:line.index('#')]

        if "def" in line and not line.endswith(')'):
            line += ')'
            flag = True

        args_list = []

        for arg in global_args:
            v = r'\b' + re.escape(arg) + r'\b'
            match = re.finditer(v, line)
            if match:
                for m in match:
                    arg_name = m.group()
                    start_index = m.start()
                    end_index = m.end()
                    if (start_index, end_index, arg_name) not in args_list:
                        args_list.append((start_index, end_index, arg_name))

        # Apply transformations
        if len(args_list) == 0:
            continue
        elif len(args_list) == 1:
            if args_list[0][2] in mapping.values():
                key = get_key(mapping, args_list[0][2])
                split_lines[j] = line[:args_list[0][0]] + key + line[args_list[0][1]:]
            else:
                new_var = random_name()
                mapping[new_var] = args_list[0][2]
                split_lines[j] = line[:args_list[0][0]] + new_var + line[args_list[0][1]:]
        else:
            # Handle multiple arguments
            args_list.sort(key=lambda t: t[0])
            new_line = ""

            if args_list[0][2] in mapping.values():
                key = get_key(mapping, args_list[0][2])
                new_line += line[:args_list[0][0]] + key + line[args_list[0][1]:args_list[1][0]]
            if args_list[0][2] not in mapping.values():
                new_var = random_name()
                mapping[new_var] = args_list[0][2]
                new_line += line[:args_list[0][0]] + new_var + line[args_list[0][1]:args_list[1][0]]

            for k in range(1, len(args_list) - 1):
                if args_list[k][2] in mapping.values():
                    key = get_key(mapping, args_list[k][2])
                    new_line += key + line[args_list[k][1]:args_list[k + 1][0]]
                else:
                    new_var = random_name()
                    mapping[new_var] = args_list[k][2]
                    new_line += new_var + line[args_list[k][1]:args_list[k + 1][0]]

            if args_list[-1][2] in mapping.values():
                key = get_key(mapping, args_list[-1][2])
                new_line += key + line[args_list[-1][1]:]
            if args_list[-1][2] not in mapping.values():
                new_var = random_name()
                mapping[new_var] = args_list[-1][2]
                new_line += new_var + line[args_list[-1][1]:]

            split_lines[j] = new_line

        if flag:
            split_lines[j] = split_lines[j][:-1]

    prompt = '\n'.join(split_lines)
    return prompt


def change_function_name_with_regex_current_segment(prompt):
    """Change function names in the current segment."""
    mapping = {}
    func_pattern = r'\bdef\s+([a-zA-Z_]\w*)\('
    split_lines = prompt.splitlines()
    global_names = set()

    for i, line in enumerate(split_lines):
        f_names = []

        # Check for function definition
        pattern = r'^\s*def\s'
        match = re.match(pattern, line)

        if match is not None:
            func_name = re.findall(func_pattern, line)
            try:
                global_names.add(func_name[0])
            except Exception as e:
                continue

            if func_name in mapping.values():
                new_name = get_key(mapping, func_name[0])
                split_lines[i] = re.sub(func_pattern, "def " + new_name + "(", line)
            else:
                new_name = random_name()
                if new_name in mapping:
                    while new_name in mapping:
                        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                                   'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
                        ran = random.choice(letters)
                        new_name = new_name + ran
                mapping[new_name] = func_name[0]
                split_lines[i] = re.sub(func_pattern, "def " + new_name + "(", line)

        # Process global function names
        for n in global_names:
            v = r'\b' + re.escape(n) + r'\b'
            matches = re.finditer(v, line)
            if matches:
                for m in matches:
                    variable_name = m.group()
                    start_index = m.start()
                    end_index = m.end()
                    if (start_index, end_index, variable_name) not in f_names:
                        f_names.append((start_index, end_index, variable_name))

        # Apply function name transformations
        if len(f_names) == 0:
            continue
        elif len(f_names) == 1:
            if f_names[0][2] in mapping.values():
                key = get_key(mapping, f_names[0][2])
                split_lines[i] = line[:f_names[0][0]] + key + line[f_names[0][1]:]
            else:
                new_var = random_name()
                mapping[new_var] = f_names[0][2]
                split_lines[i] = line[:f_names[0][0]] + new_var + line[f_names[0][1]:]
        else:
            # Handle multiple function name occurrences
            f_names.sort(key=lambda t: t[0])
            new_line = ""

            if f_names[0][2] in mapping.values():
                key = get_key(mapping, f_names[0][2])
                new_line += line[:f_names[0][0]] + key + line[f_names[0][1]:f_names[1][0]]
            if f_names[0][2] not in mapping.values():
                new_var = random_name()
                mapping[new_var] = f_names[0][2]
                new_line += line[:f_names[0][0]] + new_var + line[f_names[0][1]:f_names[1][0]]

            for k in range(1, len(f_names) - 1):
                if f_names[k][2] in mapping.values():
                    key = get_key(mapping, f_names[k][2])
                    new_line += key + line[f_names[k][1]:f_names[k + 1][0]]
                else:
                    new_var = random_name()
                    mapping[new_var] = f_names[k][2]
                    new_line += new_var + line[f_names[k][1]:f_names[k + 1][0]]

            if f_names[-1][2] in mapping.values():
                key = get_key(mapping, f_names[-1][2])
                new_line += key + line[f_names[-1][1]:]
            if f_names[-1][2] not in mapping.values():
                new_var = random_name()
                mapping[new_var] = f_names[-1][2]
                new_line += new_var + line[f_names[-1][1]:]

            split_lines[i] = new_line

    prompt = "\n".join(split_lines)

    # Apply global function name replacements
    for n in global_names:
        v = r'\b' + re.escape(n) + r'\b'
        matches = re.finditer(v, prompt)
        if matches is not None:
            key = get_key(mapping, n)
            prompt = re.sub(v, key, prompt)
            try:
                prompt = re.sub(v, key, prompt)
            except Exception:
                return prompt, mapping, []

    return prompt, mapping, global_names


def change_function_name_with_regex_other_segment(prompt, mapping, global_names):
    """Change function names in the other segment using existing mapping."""
    if global_names == []:
        return prompt

    split_lines = prompt.splitlines()

    for i, line in enumerate(split_lines):
        f_names = []

        for n in global_names:
            v = r'\b' + re.escape(n) + r'\b'
            matches = re.finditer(v, line)
            if matches:
                for m in matches:
                    variable_name = m.group()
                    start_index = m.start()
                    end_index = m.end()
                    if (start_index, end_index, variable_name) not in f_names:
                        f_names.append((start_index, end_index, variable_name))

        # Apply transformations
        if len(f_names) == 0:
            continue
        elif len(f_names) == 1:
            if f_names[0][2] in mapping.values():
                key = get_key(mapping, f_names[0][2])
                split_lines[i] = line[:f_names[0][0]] + key + line[f_names[0][1]:]
            else:
                new_var = random_name()
                mapping[new_var] = f_names[0][2]
                split_lines[i] = line[:f_names[0][0]] + new_var + line[f_names[0][1]:]
        else:
            # Handle multiple function name occurrences
            f_names.sort(key=lambda t: t[0])
            new_line = ""

            if f_names[0][2] in mapping.values():
                key = get_key(mapping, f_names[0][2])
                new_line += line[:f_names[0][0]] + key + line[f_names[0][1]:f_names[1][0]]
            if f_names[0][2] not in mapping.values():
                new_var = random_name()
                mapping[new_var] = f_names[0][2]
                new_line += line[:f_names[0][0]] + new_var + line[f_names[0][1]:f_names[1][0]]

            for k in range(1, len(f_names) - 1):
                if f_names[k][2] in mapping.values():
                    key = get_key(mapping, f_names[k][2])
                    new_line += key + line[f_names[k][1]:f_names[k + 1][0]]
                else:
                    new_var = random_name()
                    mapping[new_var] = f_names[k][2]
                    new_line += new_var + line[f_names[k][1]:f_names[k + 1][0]]

            if f_names[-1][2] in mapping.values():
                key = get_key(mapping, f_names[-1][2])
                new_line += key + line[f_names[-1][1]:]
            if f_names[-1][2] not in mapping.values():
                new_var = random_name()
                mapping[new_var] = f_names[-1][2]
                new_line += new_var + line[f_names[-1][1]:]

            split_lines[i] = new_line

    prompt = "\n".join(split_lines)

    # Apply global replacements
    for n in global_names:
        v = r'\b' + re.escape(n) + r'\b'
        matches = re.finditer(v, prompt)
        if matches is not None:
            key = get_key(mapping, n)
            prompt = re.sub(v, key, prompt)
            try:
                prompt = re.sub(v, key, prompt)
            except Exception:
                return prompt

    return prompt


# =============================================================================
# FUNCTION ELIMINATION OPERATIONS
# =============================================================================

def del_functions_bodies(code_segment, all_functions):
    """Remove function bodies and replace with summarized comments."""
    ans = extract_function_bodies_regex(code_segment)
    functions = ans[0]
    function_names = ans[1]

    for j, body in enumerate(functions):
        if all_functions == 1 and body == functions[-1]:
            break

        try:
            # Tokenize the code for the model
            tokenized_code = pythonTokenizer(body)
            # Create summary on code
            summary = pipeline([tokenized_code])[0]['summary_text']

            # Find the indentation level of the current function
            lines = body.split('\n')
            indentation = ''
            for line in lines:
                stripped_line = line.lstrip()
                if stripped_line:
                    indentation = line[:-len(stripped_line)]
                    break

            # Prepend the summary with the same indentation
            indented_summary = '\n'.join([indentation + "# " + line for line in summary.split('\n')])

            # Replace the body with the indented summary
            code_segment = code_segment.replace(body, function_names[j] + '\n' + indented_summary)

        except Exception as e:
            continue

    return code_segment


def del_functions_incremental(code_segment, other_code_segment):
    """Delete functions incrementally from code segment."""
    ans = extract_function_bodies_regex(code_segment)

    # ans[0] is all functions
    functions = ans[0]
    if len(functions) == 0:
        return code_segment, other_code_segment, {}

    function_to_del = functions[0]

    suffix_tokens_lst = ['<fim_suffix>', '<fim-suffix>', '<SUF>', '<FILL_ME>']
    mid_tokens_lst = ['<fim_middle>', '<fim-middle>', '<MID>']

    pattern = '|'.join(map(re.escape, suffix_tokens_lst))
    FIM_split = re.search(pattern, function_to_del)

    pattern_mid = '|'.join(map(re.escape, mid_tokens_lst))
    mid_in = re.search(pattern_mid, function_to_del)

    if FIM_split and mid_in:
        suffix_token = FIM_split.group()
        middle_token = mid_in.group()
        code_segment = code_segment.replace(functions[0], f" {suffix_token} {middle_token}")
    elif FIM_split:
        suffix_token = FIM_split.group()
        code_segment = code_segment.replace(functions[0], f" {suffix_token}")
    elif mid_in:
        middle_token = mid_in.group()
        code_segment = code_segment.replace(functions[0], f" {middle_token}")
    else:
        code_segment = code_segment.replace(functions[0], "")

    return code_segment, other_code_segment, {}


def del_function_body_special_tokens(code_segment, other_code_segment, all_functions):
    """Delete function bodies while preserving special tokens."""
    suffix_tokens_lst = ['<fim_suffix>', '<fim-suffix>', '<SUF>', '<FILL_ME>']
    mid_tokens_lst = ['<fim_middle>', '<fim-middle>', '<MID>']

    pattern = '|'.join(map(re.escape, suffix_tokens_lst))
    FIM_split = re.search(pattern, code_segment)

    pattern_mid = '|'.join(map(re.escape, mid_tokens_lst))
    mid_in = re.search(pattern_mid, code_segment)
    value = ""

    if FIM_split:
        # add the suffix token
        suf_token = FIM_split.group()
        idx = suffix_tokens_lst.index(suf_token)
        texts = code_segment.split(suf_token)
        a1 = del_functions_bodies(texts[0], all_functions)
        a2 = del_functions_bodies(texts[1], all_functions)
        if mid_in:
            if mid_in.group() not in a2:
                try:
                    value = a1[0] + ' ' + suf_token + a2[0] + mid_tokens_lst[idx]
                except Exception:
                    return code_segment, other_code_segment, {}
            else:
                value = a1 + ' ' + suf_token + a2
        else:
            value = a1 + ' ' + suf_token + a2
    else:
        if mid_in:
            vv = del_functions_bodies(code_segment, all_functions)
            if mid_in.group() not in vv:
                value = vv + ' ' + mid_in.group()
        else:
            value = del_functions_bodies(code_segment, all_functions)

    return value, other_code_segment, {}


def del_function_body_special_tokens_incremental(code_segment, other_code_segment):
    """Delete function bodies incrementally while preserving special tokens."""
    ans = extract_function_bodies_regex(code_segment)

    # ans[0] is all functions
    functions = ans[0]
    if len(functions) == 0:
        return code_segment, other_code_segment, {}

    i = 0
    for j in range(len(functions)):
        if len(functions[j]) == 2 and functions[j].split('\n')[1].startswith('#'):
            i += 1

    del_index = i

    function_to_del_body = functions[del_index]

    suffix_tokens_lst = ['<fim_suffix>', '<fim-suffix>', '<SUF>', '<FILL_ME>']
    mid_tokens_lst = ['<fim_middle>', '<fim-middle>', '<MID>']

    pattern = '|'.join(map(re.escape, suffix_tokens_lst))
    FIM_split = re.search(pattern, function_to_del_body)

    pattern_mid = '|'.join(map(re.escape, mid_tokens_lst))
    mid_in = re.search(pattern_mid, function_to_del_body)

    edited_function = del_functions_bodies(function_to_del_body, 0)

    if FIM_split and mid_in:
        suffix_token = FIM_split.group()
        middle_token = mid_in.group()
        code_segment = code_segment.replace(function_to_del_body, f"{edited_function} {suffix_token} {middle_token}")
    elif FIM_split:
        suffix_token = FIM_split.group()
        code_segment = code_segment.replace(function_to_del_body, f"{edited_function} {suffix_token}")
    elif mid_in:
        middle_token = mid_in.group()
        code_segment = code_segment.replace(function_to_del_body, f"{edited_function} {middle_token}")
    else:
        code_segment = code_segment.replace(function_to_del_body, f"{edited_function}")

    return code_segment, other_code_segment, {}


# =============================================================================
# CODE MODIFICATION OPERATIONS
# =============================================================================

def detect_and_replace_pii(current_code_segment, other_code_segment):
    """Detect and replace personally identifiable information in code."""
    pii_found = {
        "potential_sensitive_values": {}
    }

    # Regular expressions for different types of PII
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Sensitive keywords and pattern to capture values assigned to them
    sensitive_keywords = [
        "secret", "token", "apikey", "api_key",
        "access_token", "auth_token", "credentials", "private_key", "public_key",
        "encryption_key", "key", "license_key", "cert", "certificate", "ssh",
        "ssl", "username", "user", "login", "account", "pin", "passcode", "id",
        "identifier", "session_id", "cookie", "bearer", "oauth", "config",
        "configuration", "settings", "connection_string", "db_pass", "db_user",
        "database_url", "s3_bucket", "aws_access", "aws_secret", "gcp_key",
        "azure_key", "api_secret", "client_id", "client_secret", "fingerprint",
        "signature", "salt", "hash"
    ]

    sensitive_keywords_passwords = ["password", "passwd", "pwd"]

    value_pattern = r'({})\s*=\s*["\']?([^"\';]+)["\']?'.format(
        '|'.join(re.escape(keyword) for keyword in sensitive_keywords))
    value_pattern_passwords = r'({})\s*=\s*["\']?([^"\';]+)["\']?'.format(
        '|'.join(re.escape(keyword) for keyword in sensitive_keywords_passwords))

    # Detect and replace potential sensitive values
    matches = re.findall(value_pattern, current_code_segment, re.IGNORECASE)
    matches2 = re.findall(value_pattern_passwords, current_code_segment, re.IGNORECASE)

    for key, value in matches:
        if key.lower() in pii_found["potential_sensitive_values"]:
            pii_found["potential_sensitive_values"][key.lower()].append(value)
        else:
            pii_found["potential_sensitive_values"][key.lower()] = [value]
        # Replace the sensitive value in the code
        current_code_segment = re.sub(f'{re.escape(value)}', '<KEY>', current_code_segment)

    for key, value in matches2:
        if key.lower() in pii_found["potential_sensitive_values"]:
            pii_found["potential_sensitive_values"][key.lower()].append(value)
        else:
            pii_found["potential_sensitive_values"][key.lower()] = [value]
        # Replace the sensitive value in the code
        current_code_segment = re.sub(f'{re.escape(value)}', '<PASSWORD>', current_code_segment)

        # Detect, add, and replace emails
    pii_found["emails"] = re.findall(email_pattern, current_code_segment)
    current_code_segment = re.sub(email_pattern, '<EMAIL>', current_code_segment)

    return current_code_segment, other_code_segment, {}


def change_random_lines(code_segment, other_code_segment):
    """Randomly swap two lines in the code segment."""
    split_prompt = code_segment.splitlines()

    # prevent errors
    if len(split_prompt) < 3:
        return code_segment, other_code_segment, {}

    suffix_tokens_lst = ['<fim_suffix>', '<fim-suffix>', '<SUF>', '<FILL_ME>']
    pattern = '|'.join(map(re.escape, suffix_tokens_lst))

    # indexes are 1 and n-1 because we don't want to replace special tokens places
    n = len(split_prompt) - 1
    random_line_1_index = random.randint(1, n - 1)
    random_line_1 = split_prompt[random_line_1_index]
    random_line_2_index = random.randint(1, n - 1)
    random_line_2 = split_prompt[random_line_2_index]

    fim_split1 = re.search(pattern, random_line_1)
    fim_split2 = re.search(pattern, random_line_2)

    if fim_split1:
        for _ in range(5):
            random_line_1_index = random.randint(1, n - 1)
            random_line_1 = split_prompt[random_line_1_index]
            new_fim = re.search(pattern, random_line_1)
            if not new_fim:
                break

    if fim_split2:
        for _ in range(5):
            random_line_2_index = random.randint(1, n - 1)
            random_line_2 = split_prompt[random_line_2_index]
            new_fim = re.search(pattern, random_line_2)
            if not new_fim:
                break

    split_prompt[random_line_1_index] = random_line_2
    split_prompt[random_line_2_index] = random_line_1
    new = "\n".join(split_prompt)
    return new, other_code_segment, {}


def delete_random_line(code_segment, other_code_segment):
    """Delete a random line from the code segment."""
    suffix_tokens_lst = ['<fim_suffix>', '<fim-suffix>', '<SUF>', '<FILL_ME>']
    pattern = '|'.join(map(re.escape, suffix_tokens_lst))

    split_prompt = code_segment.splitlines()

    if len(split_prompt) < 3:
        return code_segment, other_code_segment, {}

    n = len(split_prompt) - 1
    random_line_index = random.randint(1, n - 1)
    random_line_1 = split_prompt[random_line_index]
    fim_split1 = re.search(pattern, random_line_1)
    if not fim_split1:
        split_prompt.pop(random_line_index)

    new = "\n".join(split_prompt)
    return new, other_code_segment, {}


def insert_random_line(code_segment, other_code_segment):
    """Insert a random variable assignment line into the code segment."""
    split_prompt = code_segment.splitlines()
    if len(split_prompt) < 3:
        return code_segment, other_code_segment, {}

    abc = 'abcdefghijklmnopqrstuvwxyz'
    len_code = len(split_prompt)

    # Choosing random variable, value, and insertion place
    var_ind = random.choice(abc)
    random_val = random.randint(0, 100)
    random_place = random.randint(0, len_code)

    # Determining the indentation level
    if len_code == 0:  # If the input code is empty
        indentation = ""
    elif random_place == len_code:  # If inserting at the end
        indentation = get_indentation(split_prompt[-1])
    else:
        indentation = get_indentation(split_prompt[random_place])

    # Inserting the random line at the correct indentation level
    split_prompt.insert(random_place, f"{indentation}{var_ind}={random_val}")

    new = "\n".join(split_prompt)
    return new, other_code_segment, {}


# =============================================================================
# SEGMENT-SPECIFIC WRAPPER FUNCTIONS
# =============================================================================

# Function elimination wrappers
def del_functions_incremental_segment_0(code_segment, other_code_segment):
    first_prompt, second_prompt, name_map = del_functions_incremental(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def del_functions_incremental_segment_1(code_segment, other_code_segment):
    second_prompt, first_prompt, name_map = del_functions_incremental(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def del_all_function_body_special_tokens_segment_0(code_segment, other_code_segment):
    first_prompt, second_prompt, name_map = del_function_body_special_tokens(code_segment, other_code_segment, 0)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def del_all_function_body_special_tokens_segment_1(code_segment, other_code_segment):
    second_prompt, first_prompt, name_map = del_function_body_special_tokens(code_segment, other_code_segment, 0)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def del_all_except_last_function_body_special_tokens_segment_0(code_segment, other_code_segment):
    first_prompt, second_prompt, name_map = del_function_body_special_tokens(code_segment, other_code_segment, 1)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def del_all_except_last_function_body_special_tokens_segment_1(code_segment, other_code_segment):
    second_prompt, first_prompt, name_map = del_function_body_special_tokens(code_segment, other_code_segment, 1)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def del_function_body_special_tokens_incremental_segment_0(code_segment, other_code_segment):
    first_prompt, second_prompt, name_map = del_function_body_special_tokens_incremental(code_segment,
                                                                                         other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def del_function_body_special_tokens_incremental_segment_1(code_segment, other_code_segment):
    second_prompt, first_prompt, name_map = del_function_body_special_tokens_incremental(code_segment,
                                                                                         other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


# Identifier obfuscation wrappers
def change_all_argument_names_final_segment_0(code_segment, other_code_segment):
    first_prompt, second_prompt, name_map = change_all_argument_names_final(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def change_all_argument_names_final_segment_1(code_segment, other_code_segment):
    second_prompt, first_prompt, name_map = change_all_argument_names_final(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def change_all_variables_names_final_segment_0(code_segment, other_code_segment):
    first_prompt, second_prompt, name_map = change_all_variables_names_final(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def change_all_variables_names_final_segment_1(code_segment, other_code_segment):
    second_prompt, first_prompt, name_map = change_all_variables_names_final(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def change_all_function_names_final_final_segment_0(code_segment, other_code_segment):
    first_prompt, second_prompt, name_map = change_all_function_names_final(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def change_all_function_names_final_final_segment_1(code_segment, other_code_segment):
    second_prompt, first_prompt, name_map = change_all_function_names_final(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


# Code modification wrappers
def insert_random_line_segment_0(code_segment, other_code_segment):
    first_prompt, second_prompt, name_map = insert_random_line(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def insert_random_line_segment_1(code_segment, other_code_segment):
    second_prompt, first_prompt, name_map = insert_random_line(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def delete_random_line_segment_0(code_segment, other_code_segment):
    first_prompt, second_prompt, name_map = delete_random_line(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def delete_random_line_segment_1(code_segment, other_code_segment):
    second_prompt, first_prompt, name_map = delete_random_line(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def change_random_lines_segment_0(code_segment, other_code_segment):
    first_prompt, second_prompt, name_map = change_random_lines(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def change_random_lines_segment_1(code_segment, other_code_segment):
    second_prompt, first_prompt, name_map = change_random_lines(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


# PII protection wrappers
def detect_and_replace_pii_segment_0(code_segment, other_code_segment):
    first_prompt, second_prompt, name_map = detect_and_replace_pii(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False


def detect_and_replace_pii_segment_1(code_segment, other_code_segment):
    second_prompt, first_prompt, name_map = detect_and_replace_pii(code_segment, other_code_segment)
    full_code = first_prompt + '\n' + second_prompt
    return first_prompt, second_prompt, full_code, name_map, False

