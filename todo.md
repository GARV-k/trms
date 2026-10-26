1. Is their duplication in the implementation (axis 1, axis 2 for k = 3 t= 3 should call the same code) ? 
2.   eager mode (removes gpu based minor numerical inconsistencies) ?
3. What is stochastic acl halting ?
4. How different is the current codebase with the original trm code ? 
5. Is the original trm codebase itself reproducible with same seed ?
6. Can you replace the trm arch with some other arch and then check reproducbility ? to narrow down if the bug is in arch or training pipeline.  