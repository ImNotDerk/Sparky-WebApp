from input_evaluator import InputEvaluator

def run_tests():
    evaluator = InputEvaluator()

    # --- NAME EXTRACTION TESTS ---
    print("Testing name extraction...")
    assert evaluator.extract_name("My name is Juan") == "Juan"
    assert evaluator.extract_name("my name is anna") == "Anna"
    assert evaluator.extract_name("Juan") == "Juan"
    assert evaluator.extract_name("my name is") is None
    assert evaluator.is_empty_name_phrase("my name is") is True
    assert evaluator.is_empty_name_phrase("My name is") is True

    # --- TOPIC EXTRACTION TESTS ---
    print("Testing topic extraction...")
    assert evaluator.extract_topic("I want to learn about living things") == "Living vs. Non-Living Things"
    assert evaluator.extract_topic("let’s talk about growth") == "Characteristics of Living Things - Growth"
    assert evaluator.extract_topic("plants and animals") == "Parts of Plants and Animals"
    assert evaluator.extract_topic("I want to learn about") is None
    assert evaluator.is_empty_topic_phrase("I want to learn about") is True

    print("✅ All tests passed!")


if __name__ == "__main__":
    run_tests()
