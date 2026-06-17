library(dplyr)
library(ggplot2)
library(data.table)

## Load the data ------------------------------------------
scenes_data <- fread("/Users/jorgeochoa/Library/CloudStorage/Dropbox-Brown/Jorge Ochoa/Research/building-height-prediction/data_source/data/planet_imagery/generated/cities_scenes_results_planet.csv")


## Creating some variables ----------------------------------

scenes_data[, num_day_acquired := lubridate::day(acquired)]
scenes_data[, num_month_acquired := lubridate::month(acquired)]
scenes_data[, num_year_acquired := lubridate::year(acquired)]

scenes_data[, num_day_acquired_local := lubridate::day(acquired_local)]
scenes_data[, num_month_acquired_local := lubridate::month(acquired_local)]
scenes_data[, num_year_acquired_local := lubridate::year(acquired_local)]

scenes_data[, num_hour_acquired_local := lubridate::hour(acquired_local)]

scenes_data[, date_local_month := lubridate::make_date(num_year_acquired_local, num_month_acquired_local, 1)]

## Analysis of the data ----------------------------------

scenes_data[(aoi_coverage_percent == 100)&(cloud_cover == 0)] %>%
  group_by(city) %>%
  summarise(count = n())

scenes_data[(aoi_coverage_percent == 100)&(cloud_cover == 0)] %>%
  group_by(city, date_local_month) %>%
  summarise(count = n()) %>%
  ggplot(data = .,
         aes(x = date_local_month, y = count, fill = city, group = city)) +
  geom_line(color = "black") +
  geom_point(color = "black") +
#   facet_wrap(~city, scales = "free_y", ncol = 5,nrow = 6) +
  theme_bw() +
  theme(legend.position = "none")

# Let's work with Boston
# Boston was building footprint was created in 2019. Lets filter based on that

scenes_data[(aoi_coverage_percent == 100)&(cloud_cover == 0)&(city == "Boston")] %>%
  group_by(city, date_local_month) %>%
  summarise(count = n()) %>%
  ggplot(data = .,
         aes(x = date_local_month, y = count, fill = city, group = city)) +
  geom_line(color = "black") +
  geom_point(color = "black") +
  theme_bw() +
  theme(legend.position = "none")


#scenes_data[(aoi_coverage_percent == 100) & (cloud_cover == 0) & (city == "Boston")] |> 
scenes_data[(aoi_coverage_percent == 100) & (cloud_cover == 0)] |> 
    group_by(city,num_year_acquired_local, num_month_acquired_local) |> 
    summarise(mean_cloud_cover = mean(cloud_cover),
            mean_aoi_coverage_percent = mean(aoi_coverage_percent),
            mean_sun_elevation = mean(sun_elevation),
            mean_pixel_resolution = mean(pixel_resolution),
            mean_shadow_percent = mean(shadow_percent),
            mean_snow_ice_percent = mean(snow_ice_percent),
            mean_heavy_haze_percent = mean(heavy_haze_percent),
            mean_light_haze_percent = mean(light_haze_percent)) |>
    mutate(num_year_acquired_local = as.character(num_year_acquired_local),
           char_month_acquired_local = case_when(num_month_acquired_local == 1 ~ "January",
                                                 num_month_acquired_local == 2 ~ "February",
                                                 num_month_acquired_local == 3 ~ "March",
                                                 num_month_acquired_local == 4 ~ "April",
                                                 num_month_acquired_local == 5 ~ "May",
                                                 num_month_acquired_local == 6 ~ "June",
                                                 num_month_acquired_local == 7 ~ "July",
                                                 num_month_acquired_local == 8 ~ "August",
                                                 num_month_acquired_local == 9 ~ "September",
                                                 num_month_acquired_local == 10 ~ "October",
                                                 num_month_acquired_local == 11 ~ "November",
                                                 num_month_acquired_local == 12 ~ "December"),
            char_month_acquired_local = factor(char_month_acquired_local,
                                               levels = c("January", "February", "March", "April", "May", "June",
                                                          "July", "August", "September", "October", "November", "December"))) %>%
  ggplot(data = .,
         aes(x = char_month_acquired_local, 
             y = mean_sun_elevation, 
             fill = num_year_acquired_local,
             color = num_year_acquired_local,
             group = num_year_acquired_local)) +
 geom_point() +
 facet_wrap(~city, scales = "free_y", ncol = 5,nrow = 6) +
 labs(y = "Mean Sun Elevation", color = "Year",color = "Year", fill = "Year",
 caption = "Note: Elevation angle of the sun above the horizon (degrees)") +
 theme_bw() +
 theme(legend.position = c(0.5, 0.1), axis.text.x = element_text(angle = 45, hjust = 1),
       legend.direction = "horizontal", legend.background = element_rect(color = NA, fill = NA),
       axis.title.x = element_blank(), legend.key = element_rect(color = NA, fill = NA))



## Choosing the months to analyze ---------

months_to_analyze <- c("January", "December","June","July")
scenes_data[(aoi_coverage_percent == 100) & (cloud_cover == 0) & (num_month_acquired_local  %in% c(1,12,6,7)) & (quality_category == "standard")] %>%
  filter(city == "Boston") %>%
  filter(num_year_acquired_local >= 2020) %>%
  select(city,id,acquired, cloud_cover, aoi_coverage_percent, sun_elevation, satellite_id, instrument) %>% View()
