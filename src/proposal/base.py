"""Base classes and shared templates for problem proposal strategies."""

# Base prompts for different strategies
easy_nl_base_prompt = \
'''Below is a description of a function spec in Verus (Rust). It defines the function's inputs, outputs, and returns, and may define preconditions (requires) and postconditions (ensures), and any helpers necessary for defining these attributes. This function has been given to a learner to solve, but it is easy for the learner. Please increase the difficulty of the given programming test question a bit. You may output a few different versions of the more difficult problem. You can increase the difficulty using, but not limited to, the following methods:

1. Add new constraints and requirements to the original problem, adding approximately 10 additional words.
2. Replace a commonly used requirement in the programming task with a less common and more specific one.
3. If the original problem can be solved with only a few logical steps, please add more reasoning steps.
4. Propose higher time or space complexity requirements, but please refrain from doing so frequently.

First you should reason about your answer, then you should output the problem descriptions for each problem within #### #### tags, so that it can be parsed. Each problem you propose should be STANDALONE - only that problem description will be given to a spec code generator later, which must be able to generate a spec from your natural language description alone. You should therefore use great detail in your descriptions, so that a spec can be easily created from the natural language description.

The input will look like this

# Spec description
<a description of the function spec in question, in detailed natural language>

Your solution will take the form

# Reasoning
< your observations, thoughts about what kinds of problems to propose that would be easier, and how you'd implement those problems as Verus specs >

# Output
####
first function spec you propose
####

####
second function spec you propose
####

...

####
nth function spec you propose
####

Here are some examples
'''


# Base prompts for different strategies
easy_spec_base_prompt = \
'''Given a function spec for a Verus (Rust) function that is easy for a learner to solve, I would like you to output a more difficult function spec in Verus (Rust) for our current learner to solve. A spec defines a function's inputs, outputs, and returns, and may define preconditions (requires) and postconditions (ensures), any helpers necessary for defining these attributes. You may increase the difficulty of the problem using the following strategies:
1. Combine multiple concepts, or requirements from different easy problems.
2. Add new constraints that require integrating techniques from multiple problems.
3. Create a composite problem that requires solving sub-problems similar to the easy ones as intermediate steps.
4. Increase algorithmic complexity by requiring the solver to coordinate multiple operations.
5. Add edge cases that require handling interactions between different problem aspects.

First you should reason about your answer, then you should output the problem descriptions for each problem within ```rust ``` tags, so that it can be parsed.

Your solution will take the form:

# Reasoning
< your observations about what makes the example problems easy or hard, and ideas about how to propose a new problem >

```rust
<function spec you propose>
```

Here is the function spec that is easy for the learner 
```rust
{spec}
```

Now it's your turn! Please enclose your function spec in ```rust ``` tags, so that it can be parsed. DO NOT copy the example, and DO NOT include an implementation to the function or any closing braces. Your function spec should end with an open curly brace, like the example.
'''

# Multi easy-to-hard specific prompt
multi_easy_nl_base_prompt = \
'''Below are descriptions of multiple function specs in Verus (Rust). Each defines a function's inputs, outputs, and returns, and may define preconditions (requires) and postconditions (ensures), and any helpers necessary for defining these attributes. These functions have been given to a learner to solve, and they are all easy for the learner. Please create a harder programming problem by combining concepts, techniques, or requirements from multiple of the easy problems provided. You may output a few different versions of the more difficult problem. You can increase the difficulty using, but not limited to, the following methods:

1. Combine multiple concepts, or requirements from different easy problems.
2. Add new constraints that require integrating techniques from multiple problems.
3. Create a composite problem that requires solving sub-problems similar to the easy ones as intermediate steps.
4. Increase algorithmic complexity by requiring the solver to coordinate multiple operations.
5. Add edge cases that require handling interactions between different problem aspects.

First you should reason about your answer, then you should output the problem descriptions for each problem within #### #### tags, so that it can be parsed. Each problem you propose should be STANDALONE - only that problem description will be given to a spec code generator later, which must be able to generate a spec from your natural language description alone. You should therefore use great detail in your descriptions, so that a spec can be easily created from the natural language description.

The input will look like this:

# Multiple Easy Spec Descriptions
<multiple descriptions of function specs, each in detailed natural language>

Your solution will take the form:

# Reasoning
< your observations about the easy problems, thoughts about how to combine them into harder problems, and how you'd implement those problems as Verus specs >

# Output
####
first function spec you propose
####

####
second function spec you propose
####

...

####
nth function spec you propose
####

Here are the easy functions. Please use these to create a harder problem using the strategies listed above.

Begin your reasoning by carefully analyzing the skills required to solve each easy problem, then be creative! Try to come up with problems that are distinctly new, but challenge the learner in ways that combine the concepts from the easy problems and make them harder.
'''
# Here are some examples


icl_band_base_prompt_nl = \
'''I would like you to output a function spec in Verus (Rust) that is easy, medium, hard, or impossible (specified by the prompt) for our current learner to solve. A spec defines a function's inputs, outputs, and returns, and may define preconditions (requires) and postconditions (ensures). You should also describe any helpers necessary for defining these attributes.

First you should reason about your answer, then you should output the problem descriptions for each problem within #### #### tags, so that it can be parsed.

Your solution will take the form:

# Reasoning
< your observations about what makes the example problems easy or hard, and ideas about how to propose a new problem >

####
<function spec you propose, in detailed natural language>
####

Here are some examples of problems and how difficult they were for the model.
{examples}

Now it's your turn! Please output your reasoning, then output a problem that is **{difficulty}** for the model.  Please enclose your function spec in #### #### tags, so that it can be parsed.
'''



def create_multi_easy_prompt(easy_problems, k):
    """Create a prompt with k easy problems."""
    problems_text = "\n\n".join([
        f"**Easy Problem {i+1}:**\n{problem}"
        for i, problem in enumerate(easy_problems[:k])
    ])

    prompt = f"""
# Multiple Easy Spec Descriptions
{problems_text}
"""
    return multi_easy_nl_base_prompt + prompt


hard_nl_base_prompt = \
r'''Below is a description of a function spec in Verus (Rust). It defines the function's inputs, outputs, and returns, and may define preconditions (requires) and postconditions (ensures), and any helpers necessary for defining these attributes. This function has been given to a learner to solve, but it is too challenging. Your task is to break this problem down and propose specs for easier problems that challenge the model along the same core skills, but that are easier. Your response should have the same high level of detail in natural language as the original problem description (so that a spec can be easily created from the natural language description). First you should reason about your answer, then you should output the problem descriptions for each problem within #### #### tags, so that it can be parsed. Each problem you propose should be STANDALONE - only that problem description will be given to a spec code generator later, which must be able to generate a spec from your natural language description alone. You should therefore use great detail in your descriptions, so that a spec can be easily created from the natural language description.

The input will look like this

# Spec description
<a description of the function spec in question, in detailed natural language>

Your solution will take the form

# Reasoning
< your observations, thoughts about what kinds of problems to propose that would be easier, and how you'd implement those problems as Verus specs >

# Output
####
first function spec you propose
####

####
second function spec you propose
####

...

####
nth function spec you propose
####

Here are some examples
'''


if __name__ == '__main__':
    print("Testing base...")
    
    # Test prompt templates exist and have expected content
    assert "#### ####" in easy_nl_base_prompt, "easy_nl_base_prompt format test failed"
    assert "Verus (Rust)" in easy_nl_base_prompt, "easy_nl_base_prompt content test failed"
    assert "increase the difficulty" in easy_nl_base_prompt, "easy_nl_base_prompt purpose test failed"
    
    assert "#### ####" in hard_nl_base_prompt, "hard_nl_base_prompt format test failed"
    assert "Verus (Rust)" in hard_nl_base_prompt, "hard_nl_base_prompt content test failed"
    assert "easier problems" in hard_nl_base_prompt, "hard_nl_base_prompt purpose test failed"
    
    # Test that both prompts mention standalone requirements
    assert "STANDALONE" in easy_nl_base_prompt, "easy standalone requirement test failed"
    assert "STANDALONE" in hard_nl_base_prompt, "hard standalone requirement test failed"
    
    print("✅ All base tests passed!")