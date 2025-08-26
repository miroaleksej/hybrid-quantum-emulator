# Pull Request Template

## Description

<!--- Describe your changes in detail -->
<!--- What problem does this PR solve? -->
<!--- If it fixes an open issue, please link to the issue here (e.g., Fixes #123) -->

## Related Issue

<!--- This project follows the principle: "Linear operations — in optics, non-linearities and memory — in CMOS" -->
<!--- Please specify which part of the architecture this PR targets -->
<!--- Is this related to a specific issue? Link it here: # -->

## Type of Change

<!--- Please delete options that are not relevant. -->
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires documentation update
- [ ] Performance improvement
- [ ] Platform-specific change (SOI/SiN/TFLN/InP)

## How Has This Been Tested?

<!--- Please describe the tests you ran to verify your changes. -->
<!--- Provide instructions so we can reproduce the tests. -->
<!--- Include details about your testing environment, test coverage, and the results. -->

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Documentation tests added/updated
- [ ] Manual testing completed
- [ ] Performance benchmarks completed
- [ ] Platform-specific testing (specify platforms):

```
# Example test output
```

## Benchmark Results

<!--- If this PR includes performance improvements, please provide benchmark results -->
<!--- Compare against baseline (current main branch) -->
<!--- Include metrics relevant to the project: verification speedup, memory usage, energy efficiency -->

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Verification Speedup | 1.0x | 3.64x | +264% |
| Memory Usage | 100% | 15-20% | -80-85% |
| Energy Efficiency | 100% | 145% | +45% |
| [Your metric] | [value] | [value] | [change] |

## Screenshots (if appropriate)

## Checklist

<!--- Go through the following points, and put an `x` in all the boxes that apply. -->
<!--- If you're unsure about any of these, don't hesitate to ask. We're here to help! -->

### Code Quality
- [ ] My code follows the style guidelines of this project (PEP8, type hints)
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

### Architecture Alignment
- [ ] This change aligns with the principle: "Linear operations — in optics, non-linearities and memory — in CMOS"
- [ ] This change properly separates concerns between topological analysis and photon-inspired components
- [ ] This change considers the end-to-end system (not just the core component)
- [ ] This change includes appropriate telemetry for drift monitoring
- [ ] This change considers platform-specific implementation (SOI, SiN, TFLN, InP)

### Performance & Optimization
- [ ] This change includes CPU/GPU optimization strategy
- [ ] This change considers WDM parallelism opportunities
- [ ] This change maintains or improves topological compression efficiency
- [ ] This change includes appropriate benchmarking
- [ ] This change documents expected speedup metrics

### Documentation
- [ ] I have updated the documentation accordingly
- [ ] I have added docstrings to newly created functions/methods
- [ ] I have added examples to the documentation
- [ ] I have verified documentation builds without errors

## Additional Context

<!--- Add any other context about the PR here. -->
<!--- How does this relate to the project roadmap? -->
<!--- Does this address specific pain points mentioned in issues? -->
<!--- How does this improve upon current implementation? -->
<!--- Are there any known limitations? -->

## Expected Benefits

<!--- Quantify the expected benefits: -->
<!--- How does this improve verification speedup, search speedup, or total speedup? -->
<!--- How does this impact memory usage or energy efficiency? -->
<!--- How does this improve platform compatibility or integration? -->

## Implementation Details

<!--- Provide technical details about your implementation: -->
<!--- What algorithms or techniques did you use? -->
<!--- How does this integrate with existing components? -->
<!--- Are there any performance characteristics to note? -->
<!--- Are there platform-specific considerations? -->

## Notes for Reviewers

<!--- Any specific instructions or context for reviewers -->
<!--- What parts of the code should reviewers focus on? -->
<!--- Are there any tricky parts that need extra attention? -->
<!--- How should reviewers test this change? -->

---

**Please note:**
- Good pull requests include:
  - Clear problem statement and solution
  - Comprehensive test coverage
  - Documentation updates
  - Performance metrics showing improvement
  - Alignment with the project's core principles
- For changes related to topological analysis, please specify:
  - How it affects Betti numbers calculation
  - Impact on persistent homology analysis
  - Effect on topological entropy metrics
- For photon-inspired architecture changes, please specify:
  - How it affects the interferometer grid
  - Impact on WDM parallelism
  - Effect on auto-calibration requirements
- As stated in document 2.pdf: *"Хороший PoC честно считает «всю систему», а не только красивую сердцевину из интерференции."*
- As per the roadmap: *"Как делать PoC: дорожная карта на одну страницу"*

*Thank you for contributing to the Hybrid Quantum Emulator!*
