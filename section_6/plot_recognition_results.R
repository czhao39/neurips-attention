library(plyr)
library(tidyr)

library(ggplot2)
library(ggthemes)

cbPalette <-
  c(
    "#999999",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000"
  )

# Set the appropriate directory containing results CSV files here.
setwd("~/code/attention-cogsci/section_6")

# Load the data.
df = read.csv('ANN_recognition_human_masks_results.csv')

# Renaming columns for easier reference downstream.
names(df)[names(df) == 'X1.rank.distance..correct.mask.'] <-
  'correct_rank_1'
names(df)[names(df) == 'X1.rank.distance..incorrect.mask.'] <-
  'incorrect_rank_1'
names(df)[names(df) == 'X2.rank.distance..correct.mask.'] <-
  'correct_rank_2'
names(df)[names(df) == 'X2.rank.distance..incorrect.mask.'] <-
  'incorrect_rank_2'
names(df)[names(df) == 'X3.rank.distance..correct.mask.'] <-
  'correct_rank_3'
names(df)[names(df) == 'X3.rank.distance..incorrect.mask.'] <-
  'incorrect_rank_3'
names(df)[names(df) == 'X4.rank.distance..correct.mask.'] <-
  'correct_rank_4'
names(df)[names(df) == 'X4.rank.distance..incorrect.mask.'] <-
  'incorrect_rank_4'
names(df)[names(df) == 'X5.rank.distance..correct.mask.'] <-
  'correct_rank_5'
names(df)[names(df) == 'X5.rank.distance..incorrect.mask.'] <-
  'incorrect_rank_5'

# Only keep rank of the top prediction.
df = subset(
  df,
  select = -c(
    correct_rank_2,
    incorrect_rank_2,
    correct_rank_3,
    incorrect_rank_3,
    correct_rank_4,
    incorrect_rank_4,
    correct_rank_5,
    incorrect_rank_5
  )
)

df$model <- factor(df$model)
df$mask <- factor(df$mask)

# Each dataset has a different total number of classes.
df$total_categories = mapvalues(
  df$dataset,
  from = c('CIFAR-100', 'ImageNet', 'Places365'),
  to = c(100.0, 1000.0, 434.0)
)
df$total_categories =  as.numeric(as.character(df$total_categories))

# Scale the ranks by the number of classes N in each dataset.
df$correct_rank = df$correct_rank_1 / df$total_categories
df$incorrect_rank = df$incorrect_rank_1 / df$total_categories

# Compute inverse rank via N/(N + r).
df$correct_inv_rank = 1.0 / (1.0 + df$correct_rank)
df$incorrect_inv_rank = 1.0 / (1.0 + df$incorrect_rank)

# Shorten labels.
df$attention_type = mapvalues(
  df$attention_type,
  from = c(
    'baseline-cnns',
    'attention-branch-network',
    'learn-to-pay-attention'
  ),
  to = c('base', 'abn', 'ltpa')
)

df$dataset = mapvalues(
  df$dataset,
  from = c('CIFAR-100', 'ImageNet', 'Places365'),
  to = c('CFR', 'ImgNt', 'P365')
)

# Get a unique label for (model, attention_type, dataset) triplet.
df$unique_model = paste0(df$model, '_', df$attention_type)
df$unique_model = paste0(df$unique_model, '_', df$dataset)

# Consolidate correct & incorrect ranks into one column.
df_long <-
  gather(df,
         correctORwrong,
         rank_sum,
         correct_inv_rank,
         incorrect_inv_rank,
         factor_key = TRUE)
df_long$correctORwrong <-
  factor(df_long$correctORwrong,
         levels = c('correct_inv_rank', 'incorrect_inv_rank'))


###################
##### Main plot (Figure 5).
summ  = ddply(
  df_long,
  .(mask, correctORwrong),
  summarize,
  mean = mean(rank_sum),
  ci = 1.96 * sd(rank_sum) / sqrt(length(rank_sum))
)

# Get significances.
i = 1
masks = c("KDE", "PC", "disc", "feye", "oeye", "ptch", "seye")
codes <- list()
for (id in masks) {
  temp = subset(df_long, mask == id)
  fit = lm(temp$rank_sum ~ temp$correctORwrong)
  sig = summary(fit)
  pval = sig$coefficients[8]
  pval = p.adjust(pval, "bonferroni", 7)
  if (pval < 0.001) {
    code = '***'
  }
  else if (pval < 0.01) {
    code = '**'
  }
  else if (pval < 0.05) {
    code = '*'
  }
  else {
    code = ''
  }
  codes[i] = code
  i = i + 1
}
summ$signif_code = mapvalues(summ$mask,
                             from = masks,
                             to = unlist(codes))

p <-
  ggplot(summ, aes(
    x = reorder(mask, -mean),
    y = mean,
    fill = correctORwrong
  )) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(ymin = mean - ci / 2, ymax = mean + ci / 2),
                width = .1,
                position = position_dodge(.9)) +
  geom_text(aes(x = mask, y = 0.99, label = signif_code)) +
  coord_cartesian(ylim = c(0.5, 1.0)) +
  xlab('Mask type') +
  ylab('Mean inverse rank') +
  ggtitle('N/(r+N)') +
  theme(axis.text.x = element_text(
    angle = 70,
    vjust = 1.0,
    hjust = 1.0
  )) +
  theme_hc()

p + scale_color_manual(values = cbPalette) + scale_fill_manual(values = cbPalette)

ggsave('plots/main_figure_all_masks.pdf')


###################
# More detailed analyses.

# DF with correct / wrong as a flag.
df_limited =  data.frame (
  rank  = df_long$rank_sum,
  correct_mask = df_long$correctORwrong,
  map = df_long$mask,
  model = df_long$unique_model,
  img_id = df_long$img_id
  
)

df_limited.mean <- aggregate(
  df_limited$rank,
  by = list(
    df_limited$model,
    df_limited$map,
    df_limited$correct_mask,
    df_limited$img_id
  ),
  FUN = 'mean'
)

colnames(df_limited.mean) <-
  c("model", "map", "correct_mask", "image_id", "rank")

df_limited.mean <- df_limited.mean[order(df_limited.mean$model), ]
head(df_limited.mean)

# ANOVA to check effect of map, correct/incorrect mask, and interaction
ANN_aov <- with(df_limited.mean,
                aov(rank ~ map * correct_mask +
                      Error(model / (map * correct_mask))))

s <- summary(ANN_aov)
capture.output(s, file = "plots/main_figure_anova.txt")


#######################################
# Additional analysis: Results by image (Figure S7).

# Split by img_id; average over unique model and map.
img_ids = c(
  790,
  543,
  545,
  695,
  273,
  312,
  125,
  126,
  282,
  155,
  10,
  18,
  24,
  32,
  37,
  51,
  68,
  76,
  81,
  90,
  95,
  115,
  134,
  146,
  152
)
img_id_names = c(
  'img790',
  'img543',
  'img545',
  'img695',
  'img273',
  'img312',
  'img125',
  'img126',
  'img282',
  'img155',
  'img10',
  'img18',
  'img24',
  'img32',
  'img37',
  'img51',
  'img68',
  'img76',
  'img81',
  'img90',
  'img95',
  'img115',
  'img134',
  'img146',
  'img152'
)
df_long$img_id = factor(df_long$img_id, levels = img_ids)
df_long$img_id = mapvalues(df_long$img_id, from = img_ids, to = img_id_names)

# Get significances.
i = 1
codes <- list()
img_id_code <- list()
for (id in img_id_names) {
  temp = subset(df_long, img_id == id)
  fit = lm(temp$rank_sum ~ temp$correctORwrong)
  sig = summary(fit)
  pval = sig$coefficients[8]
  pval = p.adjust(pval, "bonferroni", 25)
  if (pval < 0.001) {
    code = '***'
  }
  else if (pval < 0.01) {
    code = '**'
  }
  else if (pval < 0.05) {
    code = '*'
  }
  else {
    code = ''
  }
  codes[i] = code
  img_id_code[i] = id
  i = i + 1
}

summ  = ddply(
  df_long,
  .(img_id, correctORwrong),
  summarize,
  mean = mean(rank_sum),
  ci = 1.96 * sd(rank_sum) / sqrt(length(rank_sum))
)
summ$signif_code = mapvalues(summ$img_id, from = img_id_names, to = unlist(codes))

p <-
  ggplot(summ, aes(x = img_id, y = mean, fill = correctORwrong)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(ymin = mean - ci / 2, ymax = mean + ci / 2),
                width = .1,
                position = position_dodge(.9)) +
  geom_text(aes(x = img_id, y = 0.99, label = signif_code)) +
  coord_cartesian(ylim = c(0.6, 1.0)) +
  xlab('Image ID') +
  ylab('Mean inverse rank') +
  theme(axis.text.x = element_text(
    angle = 70,
    vjust = 1.0,
    hjust = 1.0
  )) +
  theme_hc()

p + scale_color_manual(values = cbPalette) + scale_fill_manual(values = cbPalette)

ggsave('plots/appendix_by_image.pdf')


###################
# Additional analysis: Results by model (unreported).

# Split by unique model, average over img_id and map.
summ  = ddply(
  df_long,
  .(unique_model, correctORwrong),
  summarize,
  mean = mean(rank_sum),
  ci = 1.96 * sd(rank_sum) / sqrt(length(rank_sum))
)

p <-
  ggplot(summ, aes(
    x = reorder(unique_model, -mean),
    y = mean,
    fill = correctORwrong
  )) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(ymin = mean - ci / 2, ymax = mean + ci / 2),
                width = .1,
                position = position_dodge(.9)) +
  coord_cartesian(ylim = c(0.6, 1.0)) +
  xlab('Model') + ylab('Mean inverse rank') +
  theme(axis.text.x = element_text(
    angle = 70,
    vjust = 1.0,
    hjust = 1.0
  )) +
  theme_hc()

p + scale_color_manual(values = cbPalette) + scale_fill_manual(values = cbPalette)

ggsave('plots/appendix_by_model.pdf')


#######################################
# Additional analysis (unreported).
# Split by img_id and map; averaged over unique model.

for (map.type in unique(df_long$mask)) {
  df_temp = subset(df_long, mask == map.type)
  
  # Get significances.
  i = 1
  codes <- list()
  img_id_code <- list()
  for (id in img_id_names) {
    temp = subset(df_temp, img_id == id)
    fit = lm(temp$rank_sum ~ temp$correctORwrong)
    sig = summary(fit)
    pval = sig$coefficients[8]
    pval = p.adjust(pval, "bonferroni", 25 * 4)
    if (pval < 0.001) {
      code = '***'
    }
    else if (pval < 0.01) {
      code = '**'
    }
    else if (pval < 0.05) {
      code = '*'
    }
    else {
      code = ''
    }
    codes[i] = code
    img_id_code[i] = id
    i = i + 1
  }
  
  summ  = ddply(
    df_temp,
    .(img_id, correctORwrong),
    summarize,
    mean = mean(rank_sum),
    ci = 1.96 * sd(rank_sum) / sqrt(length(rank_sum))
  )
  
  summ$signif_code = mapvalues(summ$img_id, from = img_id_names, to = unlist(codes))
  
  p <-
    ggplot(summ, aes(x = img_id, y = mean, fill = correctORwrong)) +
    geom_bar(stat = "identity", position = position_dodge()) +
    geom_errorbar(aes(ymin = mean - ci / 2, ymax = mean + ci / 2),
                  width = .1,
                  position = position_dodge(.9)) +
    geom_text(aes(x = img_id, y = 0.99, label = signif_code)) +
    coord_cartesian(ylim = c(0.6, 1.0)) +
    xlab('Image ID') +
    ylab('Mean inverse rank') +
    theme(axis.text.x = element_text(
      angle = 70,
      vjust = 1.0,
      hjust = 1.0
    )) +
    theme_hc()
  
  p + scale_color_manual(values = cbPalette) + scale_fill_manual(values = cbPalette)
  name = paste0('plots/appendix_by_image_', map.type, '.pdf')
  ggsave(name)
}

###################
# Additional analyses (unreported).

temp = subset(df_long, mask == 'PC')
fit = lm(temp$rank_sum ~ temp$correctORwrong)
summary(fit)

temp = subset(df_long, mask == 'oeye')
fit = lm(temp$rank_sum ~ temp$correctORwrong)
summary(fit)

temp = subset(df_long, mask %in% c('oeye', 'PC'))
fit = lm(temp$rank_sum ~ temp$correctORwrong + temp$mask + temp$mask * temp$correctORwrong)
summary(fit)