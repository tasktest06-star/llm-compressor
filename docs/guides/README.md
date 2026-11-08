# Guides

Welcome to the LLM Compressor guides section! Here you'll find comprehensive documentation covering key components and concepts of LLM Compressor. These guides will help you understand the various compression options available, how to apply them effectively, and how to deploy your optimized models for maximum performance.

## Key Guides

<div class="grid cards" markdown>

- :material-file-document-edit:{ .lg .middle } Recipes

    ---

    Understand how a recipe is structured — stages, modifier groups, and modifier arguments — and the rules that let a saved `recipe.yaml` be read back and reused.

    [:octicons-arrow-right-24: Recipes](recipes.md)

- :material-magnify-scan:{ .lg .middle } AWQ Scale Search

    ---

    Learn how AWQ chooses its per-channel scales, and what the `duo_scaling` and `n_grid` arguments actually sweep.

    [:octicons-arrow-right-24: AWQ Scale Search](awq_scale_search.md)

- :material-tune:{ .lg .middle } Compression Schemes

    ---

    Explore the available compression schemes for Quantization and Pruning to determine which is best for your use case.

    [:octicons-arrow-right-24: Compression Schemes](compression_schemes.md)

- :material-content-save:{ .lg .middle } Saving Models

    ---

    Learn the enhanced ways to save your compressed models with the library's extended `save_pretrained` functionality for compatibility with vLLM deployment.

    [:octicons-arrow-right-24: Saving a Model](saving_a_model.md)

- :material-content-save:{ .lg .middle } Memory requirements

    ---

    Learn about LLM Compressor's memory requirements for various supported algorithms

    [:octicons-arrow-right-24: LLM Compressor Memory Requirements](./memory.md)

- :material-content-save:{ .lg .middle } Runtime requirements

    ---

     Learn about LLM Compressor's runtime requirements for various supported algorithms

    [:octicons-arrow-right-24: LLM Compressor Memory Requirements](./runtime.md)

</div>
