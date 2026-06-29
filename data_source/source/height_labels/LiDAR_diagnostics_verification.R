library(data.table)
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)

# LiDAR diagnostics verification for choosing the building-height label metric.
# Inputs are the merged 15 x 500 with-replacement diagnostic samples produced by
# derive_lidar_building_heights.py.

HEIGHT_BINS <- c(0, 10, 20, 30, 40, 50, 100, Inf)
HEIGHT_BIN_LABELS <- c("0-10m", "10-20m", "20-30m", "30-40m", "40-50m", "50-100m", ">100m")

METRIC_MAP <- tibble::tribble(
  ~lidar_variable, ~lidar_metric,
  "height_p90_m", "LiDAR p90",
  "height_p95_m", "LiDAR p95",
  "height_max_m", "LiDAR max"
)

SUMMARY_VARIABLES <- tibble::tribble(
  ~variable, ~height_source,
  "official_height_m", "Raw height",
  "height_p90_m", "LiDAR p90",
  "height_p95_m", "LiDAR p95",
  "height_max_m", "LiDAR max"
)

clean_lidar_height <- function(height_m) {
  if_else(!is.na(height_m) & height_m > 0, height_m, NA_real_)
}

script_path <- function() {
  command_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  match <- grep(file_arg, command_args, value = TRUE)
  if (length(match) > 0) {
    path <- sub(file_arg, "", match[[1]])
    path <- gsub("~\\+~", " ", path)
    return(normalizePath(path, mustWork = TRUE))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile, mustWork = TRUE))
  }
  stop("Run this script with Rscript so the script path can be detected.", call. = FALSE)
}

PROJECT_ROOT <- normalizePath(file.path(dirname(script_path()), "../../.."), mustWork = TRUE)
OUTPUT_DIR <- file.path(
  PROJECT_ROOT,
  "data_source/data/height_labels/generated/diagnostic_analysis/metric_selection"
)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

INPUTS <- list(
  los_angeles = file.path(
    PROJECT_ROOT,
    "data_source/data/height_labels/generated/los_angeles/building_height_diagnostics_sample.csv"
  ),
  new_york_city = file.path(
    PROJECT_ROOT,
    "data_source/data/height_labels/generated/new_york_city/building_height_diagnostics_sample.csv"
  )
)

read_city_diagnostics <- function(city_slug, path) {
  if (!file.exists(path)) {
    stop("Missing diagnostic input for ", city_slug, ": ", path, call. = FALSE)
  }

  city_data <- fread(path) |>
    as_tibble() |>
    mutate(
      city_slug = city_slug,
      city_name = case_when(
        city_slug == "los_angeles" ~ "Los Angeles",
        city_slug == "new_york_city" ~ "New York City",
        TRUE ~ city_slug
      )
    )

  if (!("height_max_m" %in% names(city_data)) && "height_max_clean_m" %in% names(city_data)) {
    city_data <- city_data |>
      mutate(height_max_m = height_max_clean_m)
  }

  city_data
}

diagnostics <- bind_rows(
  read_city_diagnostics("los_angeles", INPUTS$los_angeles),
  read_city_diagnostics("new_york_city", INPUTS$new_york_city)
)

required_columns <- c(
  "city_slug",
  "city_name",
  "sample_run",
  "official_height_m",
  "height_p90_m",
  "height_p95_m",
  "height_max_m"
)
missing_columns <- setdiff(required_columns, names(diagnostics))
if (length(missing_columns) > 0) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "), call. = FALSE)
}

diagnostics_analysis <- diagnostics |>
  mutate(
    across(
      all_of(METRIC_MAP$lidar_variable),
      clean_lidar_height
    )
  )

lidar_cleaning_audit <- diagnostics |>
  select(city_slug, city_name, all_of(METRIC_MAP$lidar_variable)) |>
  pivot_longer(
    cols = all_of(METRIC_MAP$lidar_variable),
    names_to = "lidar_variable",
    values_to = "raw_lidar_height_m"
  ) |>
  left_join(METRIC_MAP, by = "lidar_variable") |>
  group_by(city_slug, city_name, lidar_variable, lidar_metric) |>
  summarise(
    total_rows = n(),
    missing_before_cleaning = sum(is.na(raw_lidar_height_m)),
    nonpositive_lidar_rows_set_to_missing = sum(!is.na(raw_lidar_height_m) & raw_lidar_height_m <= 0),
    positive_lidar_rows = sum(!is.na(raw_lidar_height_m) & raw_lidar_height_m > 0),
    .groups = "drop"
  ) |>
  arrange(city_name, lidar_metric)

write_csv(
  lidar_cleaning_audit,
  file.path(OUTPUT_DIR, "lidar_nonpositive_height_cleaning_audit.csv")
)

summary_table <- diagnostics_analysis |>
  select(city_slug, city_name, all_of(SUMMARY_VARIABLES$variable)) |>
  pivot_longer(
    cols = all_of(SUMMARY_VARIABLES$variable),
    names_to = "variable",
    values_to = "height_m"
  ) |>
  left_join(SUMMARY_VARIABLES, by = "variable") |>
  group_by(city_slug, city_name, variable, height_source) |>
  summarise(
    n = sum(!is.na(height_m)),
    missing = sum(is.na(height_m)),
    mean_m = mean(height_m, na.rm = TRUE),
    sd_m = sd(height_m, na.rm = TRUE),
    min_m = min(height_m, na.rm = TRUE),
    p01_m = quantile(height_m, 0.01, na.rm = TRUE),
    p05_m = quantile(height_m, 0.05, na.rm = TRUE),
    p25_m = quantile(height_m, 0.25, na.rm = TRUE),
    median_m = median(height_m, na.rm = TRUE),
    p75_m = quantile(height_m, 0.75, na.rm = TRUE),
    p95_m = quantile(height_m, 0.95, na.rm = TRUE),
    p99_m = quantile(height_m, 0.99, na.rm = TRUE),
    max_m = max(height_m, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(city_name, factor(height_source, levels = SUMMARY_VARIABLES$height_source))

write_csv(
  summary_table,
  file.path(OUTPUT_DIR, "raw_vs_lidar_height_summary_statistics.csv")
)

scatter_data <- diagnostics_analysis |>
  select(city_slug, city_name, building_footprint_id, sample_run, official_height_m, all_of(METRIC_MAP$lidar_variable)) |>
  pivot_longer(
    cols = all_of(METRIC_MAP$lidar_variable),
    names_to = "lidar_variable",
    values_to = "lidar_height_m"
  ) |>
  left_join(METRIC_MAP, by = "lidar_variable") |>
  filter(!is.na(official_height_m), !is.na(lidar_height_m))

city_axis_limits <- scatter_data |>
  group_by(city_slug) |>
  summarise(
    axis_min = min(c(official_height_m, lidar_height_m), na.rm = TRUE),
    axis_max = max(c(official_height_m, lidar_height_m), na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    axis_padding = pmax((axis_max - axis_min) * 0.04, 1),
    axis_min = pmin(0, axis_min - axis_padding),
    axis_max = axis_max + axis_padding
  )

for (city in unique(scatter_data$city_slug)) {
  city_limits <- city_axis_limits |>
    filter(city_slug == city)

  for (metric in METRIC_MAP$lidar_variable) {
    plot_data <- scatter_data |>
      filter(city_slug == city, lidar_variable == metric)

    city_label <- unique(plot_data$city_name)
    metric_label <- unique(plot_data$lidar_metric)
    output_name <- paste0(city, "_", metric, "_vs_raw_height_scatter.png")
    fit <- lm(official_height_m ~ lidar_height_m, data = plot_data)
    intercept <- unname(coef(fit)[[1]])
    slope <- unname(coef(fit)[[2]])
    rmse <- sqrt(mean((plot_data$lidar_height_m - plot_data$official_height_m)^2))
    equation_label <- paste0(
      "Raw = ",
      sprintf("%.2f", intercept),
      " + ",
      sprintf("%.2f", slope),
      " x LiDAR\nRMSE = ",
      sprintf("%.2f", rmse),
      " m"
    )

    plot <- ggplot(plot_data, aes(x = lidar_height_m, y = official_height_m)) +
      geom_point(alpha = 0.25, size = 1, color = "#2F5597") +
      geom_smooth(method = "lm", se = TRUE, color = "#B03A2E", linewidth = 0.8) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
      annotate(
        "label",
        x = city_limits$axis_min,
        y = city_limits$axis_max,
        label = equation_label,
        hjust = 0,
        vjust = 1,
        size = 3.4,
        fill = "white",
        alpha = 0.9
      ) +
      coord_cartesian(
        xlim = c(city_limits$axis_min, city_limits$axis_max),
        ylim = c(city_limits$axis_min, city_limits$axis_max)
      ) +
      labs(
        title = paste(city_label, metric_label, "vs. Raw Height"),
        x = paste(metric_label, "height (m)"),
        y = "Raw footprint height (m)"
      ) +
      theme_bw(base_size = 12)

    ggsave(
      filename = file.path(OUTPUT_DIR, output_name),
      plot = plot,
      width = 7,
      height = 5,
      dpi = 300
    )
  }
}

rmse_data <- scatter_data |>
  mutate(
    height_bin = cut(
      official_height_m,
      breaks = HEIGHT_BINS,
      labels = HEIGHT_BIN_LABELS,
      include.lowest = TRUE,
      right = FALSE
    ),
    squared_error = (lidar_height_m - official_height_m)^2
  ) |>
  filter(!is.na(height_bin)) |>
  group_by(city_slug, city_name, sample_run, lidar_variable, lidar_metric, height_bin) |>
  summarise(
    n = n(),
    rmse_m = sqrt(mean(squared_error, na.rm = TRUE)),
    .groups = "drop"
  ) |>
  arrange(city_name, lidar_metric, sample_run, height_bin)

write_csv(
  rmse_data,
  file.path(OUTPUT_DIR, "rmse_by_city_metric_height_bin_sample_run.csv")
)

rmse_plot <- ggplot(rmse_data, aes(x = height_bin, y = rmse_m, fill = lidar_metric)) +
  geom_boxplot(outlier.alpha = 0.5, width = 0.75) +
  facet_wrap(~city_name, ncol = 1, scales = "free_y") +
  labs(
    title = "RMSE by Raw Height Bin and LiDAR Height Metric",
    x = "Raw footprint height bin",
    y = "RMSE across 500-building sample runs (m)",
    fill = "LiDAR metric"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 35, hjust = 1)
  )

ggsave(
  filename = file.path(OUTPUT_DIR, "rmse_by_height_bin_boxplot.png"),
  plot = rmse_plot,
  width = 10,
  height = 8,
  dpi = 300
)

message("Wrote metric-selection diagnostics to: ", OUTPUT_DIR)
