Note: no pushes to this repository other than my own will be accepted. This repository's purpose is solely as a reference to my work for my bachelor thesis. <br>
The last commit to this repository that represented the work in the thesis is form the 7th of July 2026 at 22:32 local time (GMT+1). <br>
The bachelor thesis can be found [here]() <br><br>
# Known issues: 
## Non-convergence of dual problem
<ul>
<li>The dual contained a sign error causing the problem to be unbounded by default (fixed as of 15-07-2026)</li>
<li>The data presented by the NGFS scenario were meant to be read as deviations from a certain baseline, not as 
standalone data that differs from the base year 2022. (fixed as of 15-07-2026)</li>
<li>The magnitude of the variables cause a big discrepancy between values of Lagrange multipliers. For example:
The equity prices are given in order of magnitude of ten thousand, while policy rate is only given in percentages less than 10.</li>
</ul>
## Yield instead of returns for bonds
<ul>
<li>The time series representing the market returns for the bonds are instead documenting their market yield. 
Therefore it does not represent an assets price, which is necessary to determine total returns when selling and receiving interest rate</li>
</ul>
