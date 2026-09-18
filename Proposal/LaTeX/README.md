# Credit proposal LaTeX version

The proposal uses the instructor-supplied NeurIPS 2021 style. The style file is copied unchanged. The source uses the preprint option to display the group members and replaces the conference notice with the course name. The template's page dimensions, body font size and heading style are retained.

Open **Credit_Default_Proposal.tex** in TeXShop and typeset with pdfLaTeX. Typeset twice when references or table numbers change.

Alternatively, run the following command from this directory with MacTeX:

    /Library/TeX/texbin/latexmk -pdf -interaction=nonstopmode -halt-on-error Credit_Default_Proposal.tex

The resulting file is **Credit_Default_Proposal.pdf**. The compiled PDF, including references, must remain within the course's four-page proposal limit.

The matching English Markdown version is [02_Credit_Default_Proposal.md](../02_Credit_Default_Proposal.md). When changing the wording, update both the Markdown and LaTeX source before recompiling. The [Chinese guide](../04_Proposal_中文说明.md) explains the evaluation protocol and the distinction between course requirements and the conference-template examples.
