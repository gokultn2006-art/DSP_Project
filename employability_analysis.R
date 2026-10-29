# Market and employability analysis in R (companion to career_skill_gap_analysis.py)
args <- commandArgs(trailingOnly = TRUE)
dir  <- ifelse(length(args) > 0, args[1], "outputs")

sal <- read.csv(file.path(dir, "salary_data.csv"), check.names = FALSE)
skills <- setdiff(names(sal), c("log_salary", "role", "exp_level"))
sal$role <- relevel(factor(sal$role), ref = "Software Engineer")
sal$exp_level <- relevel(factor(sal$exp_level), ref = "Unknown")
f <- as.formula(paste("log_salary ~", paste(sprintf("`%s`", skills), collapse = " + "), "+ role + exp_level"))
fit <- lm(f, data = sal)
cat("Postings with salary:", nrow(sal), "  R-squared:", round(summary(fit)$r.squared, 3), "\n")
co <- summary(fit)$coefficients
rn <- rownames(co); sk <- co[match(skills, gsub("`", "", rn)), , drop = FALSE]
tab <- data.frame(skill = skills, pct_diff = round((exp(sk[, "Estimate"]) - 1) * 100, 2),
                  t_value = round(sk[, "t value"], 2), p_value = signif(sk[, "Pr(>|t|)"], 3))
tab <- tab[order(-tab$pct_diff), ]
cat("\nSkill salary differences (% change when the skill is mentioned):\n")
print(head(tab, 6), row.names = FALSE); print(tail(tab, 4), row.names = FALSE)
cat("Skills significant at 5%:", sum(tab$p_value < 0.05), "of", nrow(tab), "\n")
write.csv(tab, file.path(dir, "salary_coefficients_R.csv"), row.names = FALSE)

pr <- read.csv(file.path(dir, "pairs_data.csv"))
cat("\nResume-job pairs:", nrow(pr), "\n")
for (v in c("rd", "unw", "cos", "tfidf")) {
  ct <- cor.test(pr[[v]], pr$matched_score, method = "spearman", exact = FALSE)
  cat(sprintf("Spearman(%s, matched_score) = %.3f  (p = %.2g)\n", v, ct$estimate, ct$p.value))
}
m0 <- lm(matched_score ~ factor(jn), data = pr)
m1 <- lm(matched_score ~ factor(jn) + rd + nsk, data = pr)
cat("\nLikelihood check: does readiness add to job identity?\n")
print(anova(m0, m1))
cat("R-squared job only:", round(summary(m0)$r.squared, 3), " job + readiness + n_skills:", round(summary(m1)$r.squared, 3), "\n")

#Plot
png(file.path(dir, "r_salary_premium.png"), width = 1400, height = 900, res = 200)
sel <- rbind(head(tab, 8), tail(tab, 4))
par(mar = c(4, 9, 2, 1))
barplot(rev(sel$pct_diff), horiz = TRUE, names.arg = rev(sel$skill), las = 1, cex.names = 0.7,
        col = ifelse(rev(sel$pct_diff) < 0, "firebrick", "steelblue"), xlab = "Salary difference when skill is mentioned (%)")
abline(v = 0); dev.off()
