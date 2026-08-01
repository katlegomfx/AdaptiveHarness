# Pseudocode for the pipeline
def curate_training_pair(agent, user_prompt, history):
    # 1. Distill the messy conversation into a clean answer
    clean_answer = agent.distill_conversation(history)

    # 2. Verify quality using LLM-as-a-judge
    score = agent.verify_quality(user_prompt, clean_answer)

    # 3. Check diversity
    is_diverse = check_semantic_diversity(user_prompt)

    if score >= 4 and is_diverse:
        save_to_dataset_table(user_prompt, clean_answer)
        agent.emit("   [Dataset] High-quality QA pair saved.\n")
    else:
        agent.emit(
            f"   [Dataset] Rejected (Score: {score}, Diverse: {is_diverse}).\n")
