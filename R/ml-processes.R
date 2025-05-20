#' @include Process-class.R
#' @import gdalcubes
#' @import sf
#' @import rstac
#' @import randomForest
#' @import caret
#' @import xgboost
#' @import dplyr
#' @import readr
#' @import terra
#' @import jsonlite
#' @import stats
#' @import kernlab
#' @import reticulate
#' @import torch
#' @import abind
#' @import plumber
#' @import httr
#' @import tools
NULL


#' ml_datacube_schema
#' @description Return a list with datacube description and schema
#'
#' @return datacube list
ml_datacube_schema <- function() {
  info <- list(
    description = "A data cube with the predicted values. It removes the specified dimensions and adds new dimension for the predicted values.",
    schema = list(type = "object", subtype = "raster-cube")
  )
  return(info)
}

#' ml_model_schema
#' @description Return a list with datacube description and schema
#'
#' @return model list
ml_model_schema <- function() {
  info <- list(
    description = "A ML model that was trained with one of the ML training processes.",
    schema = list(type = "object", subtype = "mlm-model")
  )
  return(info)
}

#' return objects for the processes
eo_datacube <- ml_datacube_schema()
ml_model <- ml_model_schema()

#############################################################################
# Read the environment variable "SHARED_TEMP_DIR" (returns "" if not set)
shared_dir <- Sys.getenv("SHARED_TEMP_DIR")
if (shared_dir == "") {
  shared_dir <- file.path(getwd(), "shared_temp")
  if (!dir.exists(shared_dir)) {
    dir.create(shared_dir, recursive = TRUE)
    message("Shared directory automatically created: ", normalizePath(shared_dir))
  } else {
    message("Using existing shared directory: ", normalizePath(shared_dir))
  }
  Sys.setenv(SHARED_TEMP_DIR = shared_dir)
} else {
  message("Using shared directory from environment: ", normalizePath(shared_dir))
}

ensure_python_env <- function(required_modules = c("numpy", "onnxruntime", "torch")) {
  library(reticulate)

  py_valid <- tryCatch({
    py_path <- py_config()$python
    if (!file.exists(py_path)) stop("Python binary not found.")
    TRUE
  }, error = function(e) FALSE)

  if (py_valid) {
    message("Gültige Python-Umgebung erkannt: ", py_config()$python)
    use_virtualenv("r-reticulate", required = TRUE)
  } else {
    message("Keine gültige Python-Umgebung gefunden – automatischer Aufbau beginnt...")

    if (!miniconda_exists()) {
      message("Miniconda wird installiert...")
      install_miniconda()
    }

    if ("r-reticulate" %in% virtualenv_list()) {
      message("Alte r-reticulate-Umgebung wird entfernt...")
      virtualenv_remove("r-reticulate", confirm = FALSE)
    }

    message("Neue Python-Umgebung wird erstellt...")
    virtualenv_create("r-reticulate", python = "python3")
    use_virtualenv("r-reticulate", required = TRUE)
  }

  for (mod in required_modules) {
    if (!py_module_available(mod)) {
      message("Installiere fehlendes Python-Modul: ", mod)
      py_install(mod, pip = TRUE)
    }
  }

  message("Python-Setup abgeschlossen: ", py_config()$python)
}





#############################################################################
##### ml-processes #####

##### preprocess_training_set #####
#' Preprocess the training dataset for ML training
#'
#' @description
#' Prepares training data by aligning coordinate reference systems (CRS),
#' extracting features from the raster data cube, and formatting the result
#' for machine-learning workflows.
#'
#' @param training_set Character. Path to the training data file (GeoJSON or CSV).
#' @param aot_cube Object of subtype "raster-cube". The data cube used for feature extraction.
#' @param target_column Character. Name of the target variable column in the training data.
#' @param multiple_time_steps Logical. Whether to consider multiple time steps (TRUE) or only one (FALSE). Default: FALSE.
#' @param is_classification Logical or NULL. If TRUE, treat as classification; if FALSE, as regression; if NULL, auto-detect based on target variable. Default: NULL.
#' @param srs_train_data Integer or FALSE. EPSG code to assign as default CRS if the training data lacks one. Default: FALSE.
#'
#' @return
#' A data.frame containing the processed training samples, with spatial features
#' extracted and aligned to the raster cube, ready for downstream model training.
#'
preprocess_training_set <- Process$new(
  id = "preprocess_training_set",
  description = "Prepares training data by aligning CRS, extracting data from the cube, and formatting features.",
  categories = as.array("data-preprocessing", "machine-learning"),
  summary = "Preprocess the training dataset for ML-training",

  parameters = list(
    Parameter$new(
      name        = "training_set",
      description = "Path to the training data file (GeoJSON, CSV or Shapefile)",
      schema      = list(type = "string")
    ),
    Parameter$new(
      name = "aot_cube",
      description = "The data cube used for feature extraction",
      schema = list(type = "object", subtype = "raster-cube")
    ),
    Parameter$new(
      name = "target_column",
      description = "The name of the target variable column",
      schema = list(type = "string")
    ),
    Parameter$new(
      name = "multiple_time_steps",
      description = "Whether to consider multiple time steps (TRUE) or just one (FALSE).",
      schema = list(type = "boolean"),
      optional = TRUE
    ),
    Parameter$new(
      name = "is_classification",
      description = "Whether the task is a classification problem (TRUE) or regression (FALSE). If NULL, it will be determined automatically.",
      schema = list(type = "boolean"),
      optional = TRUE
    ),
    Parameter$new(
      name = "srs_train_data",
      description = "Default CRS for training data if not specified",
      schema = list(type = "integer"),
      optional = TRUE
    )
  ),
  returns = list(
    description = "Processed training data aligned with the raster cube.",
    schema = list(type = "object")),

  operation = function(
    training_set,
    aot_cube,
    target_column,
    multiple_time_steps = FALSE,
    is_classification = NULL,
    srs_train_data = FALSE,
    job
  ) {

    ###############################
    extract_crs_datacube <- function(aot_cube){
      cube_crs <- gdalcubes::srs(aot_cube)
      crs_data <- as.numeric(gsub("EPSG:", "", cube_crs))
      message(paste("CRS of the data cube:", crs_data))
      return(crs_data)
    }

    # Transforms the training data into the coordinate system of the data cube
    transform_training_data <- function(train_dat, aot_crs){
      train_dat <- sf::st_transform(train_dat, crs = aot_crs)
      message("Training data transformed:")
      return(train_dat)
    }

    get_train_data <- function(file_path, default_crs = FALSE) {
      if (grepl("\\.geojson$", file_path, ignore.case = TRUE)) {
        message("Load GeoJSON-Data...")
        training_data <- sf::read_sf(file_path)
        message("GeoJSON-Daten geladen")

      } else if (grepl("\\.shp$", file_path, ignore.case = TRUE)) {
        message("Load Shapefile-Data...")
        training_data <- sf::read_sf(file_path)
        message("Shapefile-Daten geladen")

        # analog wie GeoJSON: falls kein CRS vorhanden, Standard-CRS setzen
        if (is.na(sf::st_crs(training_data))) {
          if (!isFALSE(default_crs)) {
            message(paste("Set default CRS EPSG:", default_crs))
            sf::st_crs(training_data) <- default_crs
          } else {
            stop("Shapefile hat kein CRS und es wurde kein Default-CRS angegeben.")
          }
        }

        # fid-Spalte ergänzen, falls nötig
        if (!"fid" %in% colnames(training_data)) {
          message("Create a unique `fid` column for Shapefile...")
          training_data$fid <- seq_len(nrow(training_data))
        } else {
          training_data$fid <- as.integer(training_data$fid)
        }

      } else if (grepl("\\.csv$", file_path, ignore.case = TRUE)) {
        # ... dein bestehender CSV-Zweig ...
      } else {
        stop("File format not supported. Please use GeoJSON, CSV or Shapefile.")
      }

      # Ensure geometry column is last
      training_data <- training_data %>%
        select(-geometry, everything(), geometry)

      return(training_data)
    }



    data_preprocessing_single <- function(aot_cube, train_dat){
      message("Starting data extraction...")

      if (!"fid" %in% colnames(train_dat)) {
        stop("The column `fid` is missing in the training data.")
      }
      train_dat$fid <- as.integer(train_dat$fid)

      extraction <- gdalcubes::extract_geom(
        cube = aot_cube,
        sf = train_dat
      )
      message("Extraction result:")

      if (nrow(extraction) == 0) {
        stop("No data extracted. Check if the bounding boxes of the training data and the data cube overlap.")
      }
      predictors_name <- get_cube_band_names(aot_cube)
      train_dat$PolyID <- seq_len(nrow(train_dat))
      extraction <- base::merge(extraction, train_dat, by.x = "FID", by.y = "PolyID")
      message("Extraction merged with training data ....")

      train_ids <- caret::createDataPartition(extraction$FID, p = 0.2, list = FALSE)
      train_data <- extraction[train_ids, ]
      train_data <- train_data[stats::complete.cases(train_data[, predictors_name]), ]
      train_data <- base::as.data.frame(train_data)
      message("Training data prepared . . . .")
      return(train_data)
    }

    data_preprocessing_multiple <- function(aot_cube, train_dat) {
      message("Starting data extraction for multiple time steps...")
      if (!"fid" %in% colnames(train_dat)) {
        stop("The column `fid` is missing in the training data")
      }
      train_dat$fid <- as.integer(train_dat$fid)  # `fid` als Integer sicherstellen

      extraction <- gdalcubes::extract_geom(
        cube = aot_cube,
        sf = train_dat
      )
      unique_times <- unique(extraction$time)
      message(paste("You have", length(unique_times), "time steps:"))
      message(unique_times)

      if (nrow(extraction) == 0) {
        stop("No data extracted. Check if the bounding boxes of the training data and the data cube overlap.")
      }

      train_dat$PolyID <- seq_len(nrow(train_dat))
      extraction <- merge(extraction, train_dat, by.x = "FID", by.y = "PolyID", all.x = TRUE)
      extraction$pixel_id <- seq_len(nrow(extraction))
      extraction$time_numeric <- as.numeric(as.Date(extraction$time))

      return(extraction)
    }


    convert_to_wide_format <- function(train_data, band_names) {
      time_steps <- unique(train_data$time)
      n_steps <- length(time_steps)

      data_frames <- list()

      has_ndvi <- "NDVI" %in% colnames(train_data)

      for (i in seq_along(time_steps)) {
        data_time <- train_data %>%
          filter(time == time_steps[i]) %>%
          dplyr::select(-geometry) %>%
          dplyr::rename_with(
            ~ paste0(., "_T", i),
            all_of(c(band_names, if (has_ndvi) "NDVI"))
          ) %>%
          dplyr::select(FID, pixel_id, starts_with("B"), if (has_ndvi) starts_with("NDVI"), -time, -time_numeric)

        data_time$pixel_id <- seq_len(nrow(data_time))

        data_frames[[i]] <- data_time
        message(paste("Rename data for time step", i))
      }

      wide_data <- Reduce(function(x, y) inner_join(x, y, by = c("FID", "pixel_id")), data_frames)

      complete_data <- wide_data[complete.cases(wide_data), ]

      message("Data in wide format after merging the time steps:")

      return(complete_data)
    }



    get_cube_band_names <- function(cube) {
      band_names <- names(cube)
      return(band_names)
    }

    time_steps_query <- function(cube){
      time_steps <- gdalcubes::dimension_values(cube)
      time_steps <- time_steps$t
      return(time_steps)
    }


    filter_polygons_within_cube_extent <- function(train_dat, aot_cube, buffer = 0) {
      cube_dims <- gdalcubes::dimensions(aot_cube)

      cube_bbox <- sf::st_bbox(c(
        xmin = cube_dims$x$low,
        xmax = cube_dims$x$high,
        ymin = cube_dims$y$low,
        ymax = cube_dims$y$high
      ), crs = sf::st_crs(train_dat))

      bbox_poly <- sf::st_as_sfc(cube_bbox)

      if (buffer != 0) {
        bbox_poly <- sf::st_buffer(bbox_poly, dist = -abs(buffer))
      }

      n_before <- nrow(train_dat)

      train_dat_filtered <- sf::st_filter(train_dat, bbox_poly, .predicate = sf::st_within)

      n_after <- nrow(train_dat_filtered)
      n_removed <- n_before - n_after

      message(paste(n_after, "Keep polygons within the data cube"))
      message(paste(n_removed, "Polygons have been removed (outside or border areas)"))

      if (n_after == 0) {
        stop("No valid training data found in the cube area (after filtering).")
      }

      return(train_dat_filtered)
    }


    add_fid_if_missing <- function(sf_obj) {
      if (! "fid" %in% names(sf_obj)) {
        sf_obj$fid <- seq.int(nrow(sf_obj))
      }
      return(sf_obj)
    }



    ###############################

    message("The data is prepared by the preprocess_training_set...")
    time <- time_steps_query(aot_cube)
    message(time)
    training_set <- get_train_data(training_set, srs_train_data)

    crs_data <- extract_crs_datacube(aot_cube)
    train_dat <- transform_training_data(train_dat = training_set, aot_crs = crs_data)
    message(train_dat)
    train_dat <- filter_polygons_within_cube_extent(train_dat, aot_cube, buffer = 10)


    y <- train_dat %>%
      sf::st_set_geometry(NULL) %>%
      dplyr::pull(!!rlang::sym(target_column))

    if (is.null(is_classification)) {
      is_classification <- is.factor(y) || is.character(y)
      message(if (is_classification) "Classification detected." else "Regression detected.")
    } else {
      message(if (is_classification) "Explicit classification set." else "Explicit regression set")
    }

    if (is_classification && !is.factor(y)) {
      message("Numerical target variable is converted into a factor.")
      train_dat[[target_column]] <- as.factor(train_dat[[target_column]])
    }

    train_data <- add_fid_if_missing(train_dat)

    if (multiple_time_steps) {
      train_data <- data_preprocessing_multiple(aot_cube = aot_cube, train_dat = train_dat)
      band_names <- get_cube_band_names(aot_cube)
      train_data_wide <- convert_to_wide_format(train_data = train_data, band_names = band_names)

      colnames(train_data_wide)[colnames(train_data_wide) == "FID"] <- "fid"
      train_data <- merge(
        train_data_wide,
        train_dat %>%
          st_set_geometry(NULL) %>%
          dplyr::select(fid, !!rlang::sym(target_column)),
        by = "fid",
        all.x = TRUE
      )
    } else {
      train_data <- data_preprocessing_single(aot_cube = aot_cube, train_dat = train_dat)
      train_data_no_geom <- train_data %>%
        dplyr::select(-geometry)

      train_data_no_geom <- train_data_no_geom[complete.cases(train_data_no_geom), ]

      geometry_data <- train_dat %>%
        dplyr::select(fid, geometry)

      train_data <- merge(train_data_no_geom, geometry_data, by = "fid", all.x = TRUE)
    }

    return(train_data)
  }
)

########################################################

##### ml_fit #####
#' Train an ML or DL model on extracted features
#'
#' @description
#' Trains a Machine Learning or Deep Learning model based on the provided
#' preprocessed training data and specified target variable.
#'
#' @param model Object of subtype "mlm-model". The machine learning or deep learning model to train.
#' @param training_set Object of subtype "vector". Preprocessed training dataset containing features and target.
#' @param target_column Character. Name of the column in `training_set` that contains the target variable.
#'
#' @return
#' A trained model object of subtype "mlm-model", ready for predictions.
#'
ml_fit <- Process$new(
  id = "ml_fit",
  description = "Trains a Machine Learning or Deep Learning model based on input training data.",
  categories = as.array("machine-learning", "model-training"),
  summary = "Train an ML/DL model on extracted features.",

  parameters = list(
    Parameter$new(
      name = "model",
      description = "The machine learning model to train",
      schema = list(type = "object", subtype = "mlm-model")
    ),
    Parameter$new(
      name = "training_set",
      description = "Preprocessed training dataset for the model",
      schema = list(type = "object", subtype = "vector")
    ),
    Parameter$new(
      name = "target_column",
      description = "The column containing the target variable",
      schema = list(type = "string")
    )
  ),

  returns = list(
    description = "Trained model",
    schema = list(type = "object", subtype = "mlm-model")
  ),
  operation = function(model, training_set, target_column, job) {

    ############# help functions ##############
    extract_features <- function(data, pattern = "_T\\d+$", bands = NULL, label_column = NULL) {
      feature_names <- colnames(data)

      if (!is.null(bands)) {
        time_step_columns <- grep(pattern, feature_names, value = TRUE)
        unique_steps <- unique(gsub(".*_T", "", time_step_columns))
        if (length(unique_steps) == 0) {
          unique_steps = 1
        }

        label_count <- if (!is.null(label_column)) {
          length(unique(data[[label_column]]))
        } else {
          NA
        }

        return(list(
          band = bands,
          time_steps = length(unique_steps),
          label_count = label_count
        ))
      }

      is_band <- function(name) {
        grepl("^(B\\d{2}|NDVI)$", name, ignore.case = TRUE)
      }

      if (any(grepl(pattern, feature_names))) {
        time_step_columns <- grep(pattern, feature_names, value = TRUE)

        unique_steps <- unique(gsub(".*_T", "", time_step_columns))

        relevant_features <- unique(gsub(pattern, "", time_step_columns))

        relevant_features <- relevant_features[sapply(relevant_features, is_band)]

        label_count <- if (!is.null(label_column)) {
          length(unique(data[[label_column]]))
        } else {
          NA
        }

        return(list(
          band = relevant_features,
          time_steps = length(unique_steps),
          label_count = label_count
        ))
      } else {
        relevant_features <- feature_names[sapply(feature_names, is_band)]

        label_count <- if (!is.null(label_column)) {
          length(unique(data[[label_column]]))
        } else {
          NA
        }

        return(list(
          band = relevant_features,
          time_steps = 1,
          label_count = label_count
        ))
      }
    }


    identify_predictors <- function(training_set, pattern = "^(B\\d+|(?i)NDVI(_T\\d+)?)$") {
      predictor_names <- colnames(training_set)
      predictor_names <- predictor_names[
        grepl(pattern, predictor_names) &
          sapply(training_set[, predictor_names, drop = FALSE], is.numeric)
      ]
      if (length(predictor_names) == 0) {
        stop("No valid predictors detected. Please check.")
      }
      return(predictor_names)
    }

    extract_time_series_features <- function(training_set, features_data, time_steps) {
      if (time_steps > 1) {
        message("multi time steps")
        features <- array(
          data = as.matrix(training_set[, grep("_T\\d+$", colnames(training_set))]),
          dim = c(nrow(training_set), length(features_data), time_steps)
        )
      } else {
        message("one time step")
        features <- array(
          data = as.matrix(training_set[, features_data]),
          dim = c(nrow(training_set), length(features_data), time_steps)
        )
      }
      return(features)
    }

    time_steps_query <- function(cube){
      time_steps <- gdalcubes::dimension_values(cube)
      time_steps <- time_steps$t
      return(time_steps)
    }


    ##################################################################################
    message("ml_fit is being prepared...")
    message("jo was")

    if (!is.null(model$parameters$cnn_layer)) {
      message("Deep learning model recognized. Start DL calculation...")

      extracted_data <- extract_features(training_set, label_column = target_column)

      features_data <- extracted_data$band
      time_steps <- extracted_data$time_steps
      class_count <- extracted_data$label_count


      features <- extract_time_series_features(training_set, features_data, time_steps)

      library(torch)


      labels <- as.numeric(as.factor(training_set[[target_column]]))
      x_train <- torch::torch_tensor(features, dtype = torch_float())
      y_train <- torch::torch_tensor(labels, dtype = torch_long())
      print(time_steps)

      dl_model <- model$create_model(
        input_data_columns = features_data,
        time_steps = time_steps,
        class_count = class_count
      )

      optimizer <- optim_adam(dl_model$parameters, lr = model$parameters$learning_rate, weight_decay = model$parameters$weight_decay)
      loss_fn <- nn_cross_entropy_loss()

      for (epoch in 1:model$parameters$epochs) {
        dl_model$train()
        optimizer$zero_grad()
        predictions <- dl_model(x_train)
        loss <- loss_fn(predictions, y_train)
        loss$backward()
        optimizer$step()
        print(sprintf("Epoch: %d, Loss: %.4f", epoch, loss$item()))
      }
      dl_model$eval()
      with_no_grad({
        predictions <- dl_model(x_train)
        predicted_classes <- torch_argmax(predictions, dim = 2)
        accuracy <- mean(as.numeric(predicted_classes == y_train))
        message(sprintf("Accuracy: %.2f%%", accuracy * 100))
      })
      confusion_matrix <- table(Predicted = as.numeric(predicted_classes), Actual = as.numeric(y_train))
      message(confusion_matrix)

      model_file <- tempfile(fileext = ".pt")
      torch_save(dl_model, model_file)
      message("Model saved in Torch file: ", model_file)

      return(model_file)
    }

    message("Machine learning model recognized. Start ML calculation...")

    y <- training_set[[target_column]]
    if (!is.numeric(y)) {
      y <- as.factor(y)
      message("Classification is carried out...")
    } else {
      y <- as.numeric(y)
      message("Regression is carried out...")
    }

    predictor_names <- identify_predictors(training_set)


    if (length(predictor_names) == 0) {
      stop("No valid predictors detected. Please check.")
    }
    message("Automatically recognized predictors: ", paste(predictor_names, collapse = ", "))

    x <- as.data.frame(lapply(training_set[, predictor_names, drop = FALSE], as.numeric))

    if (ncol(x) == 0) {
      stop("No predictors detected.")
    }

    if (model$method == "rf") {
      if (is.null(model$tuneGrid) || is.na(model$tuneGrid$mtry)) {
        max_variables <- max(1, floor(sqrt(ncol(x))))
        model$tuneGrid <- expand.grid(mtry = max_variables)
        message("tuneGrid was automatically set to mtry = ", max_variables)
      } else if (model$tuneGrid$mtry < 1 || model$tuneGrid$mtry > ncol(x)) {
        warning("Invalid `mtry` value in tuneGrid. Set `mtry` to 1.")
        model$tuneGrid <- expand.grid(mtry = 1)
      }
    } else if (model$method %in% c("svmRadial", "svmLinear", "svmPoly")) {
      if (is.null(model$tuneGrid)) {
        stop("SVM models require a defined tuneGrid")
      }
      model$preProcess <- c("center", "scale")
    } else if (model$method == "xgbTree") {
      if (is.null(model$tuneGrid)) {
        stop("XGBoost model require a defined tuneGrid")
      }
      if (is.null(model$trControl)) {
        model$trControl <- caret::trainControl(method = "cv", number = 5, search = "grid")
      }
    } else {
      stop("Undetected method! You can only work with: 'rf', 'svmRadial', 'svmLinear', 'svmPoly', 'xgbTree'.")
    }

    model <- caret::train(
      x = x,
      y = y,
      method = model$method,
      tuneGrid = model$tuneGrid,
      trControl = model$trControl,
      ntree = if (model$method == "rf") model$ntree else NULL,
      preProcess = if (model$method %in% c("svmRadial", "svmLinear", "svmPoly")) model$preProcess else NULL
    )

    if (!is.numeric(y)) {
      if ("Accuracy" %in% colnames(model$results)) {
        accuracy <- max(model$results$Accuracy, na.rm = TRUE)
        message("Accuracy: ", round(accuracy * 100, 2), "%")
      }
    }
    return(model)
  }
)


########################################################
##### ml_predict #####
#' Apply a trained ML/DL model to a data cube and return predictions
#'
#' @description
#' Applies a machine learning or deep learning model to every pixel (and/or time step)
#' of the input raster-cube and returns a new data cube with the predicted values.
#'
#' @param cube Object of subtype "raster-cube". The area-of-interest data cube on which predictions will be made.
#' @param model Object of subtype "mlm-model" or a file path (ONNX, .rds, or .pt). The trained model to use for prediction.
#'
#' @return
#' A raster-cube object containing the predicted values in a new "prediction" band.
#'
ml_predict <- Process$new(
  id = "ml_predict",
  description = "Applies a machine learning model to a data cube and returns the predicted values.",
  categories = as.array("machine-learning", "prediction"),
  summary = "Predicts the aoi (area of interest) with the model",

  parameters = list(
    Parameter$new(
      name = "cube",
      description = "The Area of interest cube, which we want to predict",
      schema = list(type = "objects", subtype = "raster-cube")
    ),
    Parameter$new(
      name = "model",
      description = "The trained machine learning model (ONNX, RDS or Model-data)",
      schema = list(type = "object", subtype ="mlm-model")
    )

  ),
  returns = eo_datacube,
  operation = function(cube, model, job) {
    ################################################################################################
    message("TestF")

    is_torch_model <- function(model) {
      inherits(model, "nn_module") ||
        "nn_module" %in% class(model) ||
        (!is.null(model$conv_layers) && !is.null(model$dense))
    }

    mlm_single <- function(data_cube, model) {
      message("Preparing prediction with apply_pixel using temporary directory...")
      tmp <- Sys.getenv("SHARED_TEMP_DIR", tempdir())

      if (is.character(model)) {
        message("Model is a file path: ", model)
        model_file <- model
      } else {
        if (is_torch_model(model)) {
          model_file <- tempfile(fileext = ".pt")
          torch::torch_save(model, model_file)
        } else {
          model_file <- file.path(tmp, "model.rds")
          saveRDS(model, model_file)
          message("Classic ML model saved to RDS: ", model_file)
        }
      }
      Sys.setenv(MODEL_FILE = model_file)

      cube_band_names <- gdalcubes::bands(data_cube)$name
      band_names_file <- file.path(tmp, "band_names.rds")
      tryCatch({
        saveRDS(cube_band_names, band_names_file)
        message("Band name file saved as: ", normalizePath(band_names_file))
      }, error = function(e) {
        stop("Error when saving the band names: ", e$message)
      })

      time_steps <- gdalcubes::dimension_values(data_cube)$t
      nsteps <- length(time_steps)
      Sys.setenv(TMPDIRPATH = tmp)
      Sys.setenv(NSTEPS = nsteps)


      predict_pixel_fun <- function(x) {
        local_tmp <- Sys.getenv("TMPDIRPATH")
        nsteps <- as.numeric(Sys.getenv("NSTEPS"))
        model_file <- Sys.getenv("MODEL_FILE")

        is_torch_model <- function(model) {
          inherits(model, "nn_module") ||
            "nn_module" %in% class(model) ||
            (!is.null(model$conv_layers) && !is.null(model$dense))
        }

        if (!is.matrix(x)) {
          x <- matrix(x, nrow = 1)
        }

        local_bands <- readRDS(file.path(local_tmp, "band_names.rds"))
        if (is.null(colnames(x))) {
          colnames(x) <- local_bands
        }

        pixel_df <- as.data.frame(x)

        library(torch)
        if (endsWith(model_file, ".pt")) {
          local_model <- torch::torch_load(model_file)
        } else {
          local_model <- readRDS(model_file)
        }

        if (is_torch_model(local_model)) {
          message("Deep Learning model detected in predict_pixel_fun")
          pixel_matrix <- as.matrix(pixel_df)
          n_channels <- length(local_bands)
          pixel_tensor <- torch::torch_tensor(pixel_matrix, dtype = torch_float())
          pixel_tensor <- pixel_tensor$view(c(1, n_channels, 1))
          message("Shape of pixel_tensor: ", paste(dim(pixel_tensor), collapse = " x "))
          local_model$eval()
          with_no_grad({
            preds <- local_model(pixel_tensor)
          })
          pred_class_tensor <- torch::torch_argmax(preds, dim = 2)
          pred_class <- as.numeric(torch::as_array(pred_class_tensor))
          return(pred_class)
        } else {
          message("Classic ML model used in predict_pixel_fun")
          pred <- predict(local_model, newdata = pixel_df)
          return(as.numeric(pred))
        }
      }

      prediction_cube <- tryCatch({
        apply_pixel(
          data_cube,
          names = "prediction",
          keep_bands = FALSE,
          FUN = predict_pixel_fun
        )
      }, error = function(e) {
        message("Error during apply_pixel: ", conditionMessage(e))
        NULL
      })

      return(prediction_cube)
    }



    mlm_multi <- function(data_cube, model) {
      message("Preparing prediction with apply_time using temporary directory...")
      tmp <- Sys.getenv("SHARED_TEMP_DIR", tempdir())

      if (is.character(model)) {
        model_file <- model
      } else {
        if (is_torch_model(model)) {
          model_file <- tempfile(fileext = ".pt")
          torch::torch_save(model, model_file)
        } else {
          model_file <- file.path(tmp, "model.rds")
          saveRDS(model, model_file)
          message("Classic ML model saved to RDS: ", model_file)
        }
      }

      Sys.setenv(MODEL_FILE = model_file)

      cube_band_names <- gdalcubes::bands(data_cube)$name
      band_names_file <- file.path(tmp, "band_names.rds")
      tryCatch({
        saveRDS(cube_band_names, band_names_file)
      }, error = function(e) {
        stop("Error when saving the band names: ", e$message)
      })
      Sys.setenv(TMPDIRPATH = tmp)

      time_steps <- gdalcubes::dimension_values(data_cube)$t
      nsteps <- length(time_steps)
      Sys.setenv(NSTEPS = nsteps)
      message("Number of time steps: ", nsteps)


      predict_time_fun <- function(x) {
        local_tmp <- Sys.getenv("TMPDIRPATH")
        local_nsteps <- as.numeric(Sys.getenv("NSTEPS"))
        model_file <- Sys.getenv("MODEL_FILE")
        is_torch_model <- function(model) {
          inherits(model, "nn_module") ||
            "nn_module" %in% class(model) ||
            (!is.null(model$conv_layers) && !is.null(model$dense))
        }


        if (!is.matrix(x)) {
          x <- matrix(x, nrow = length(readRDS(file.path(local_tmp, "band_names.rds"))))
        }

        local_bands <- readRDS(file.path(local_tmp, "band_names.rds"))
        n_bands <- length(local_bands)
        if (nrow(x) != n_bands || ncol(x) != local_nsteps) {
          stop("The dimensions of the pixel time series do not match: Expected ",
               n_bands, " Lines and ", local_nsteps, " Columns, but preserved: ",
               nrow(x), " x ", ncol(x))
        }



        library(torch)
        if (endsWith(model_file, ".pt")) {
          local_model <- torch::torch_load(model_file)
        } else {
          local_model <- readRDS(model_file)
        }

        if(is_torch_model(local_model)){
          message("Deep Learning model detected")
          pixel_tensor <- torch::torch_tensor(x, dtype = torch_float())
          pixel_tensor <- pixel_tensor$view(c(1, n_bands, local_nsteps))

          local_model$eval()
          with_no_grad({
            preds <- local_model(pixel_tensor)
          })
          pred_class_tensor <- torch::torch_argmax(preds, dim = 2)
          pred_class <- as.numeric(torch::as_array(pred_class_tensor))
          return(matrix(rep(pred_class, local_nsteps), nrow = 1))
        }else{
          message("Classic machine learning detected")
          wide_vec <- as.vector(x)
          wide_mat <- matrix(wide_vec, nrow = 1)

          wide_names <- unlist(lapply(seq_len(local_nsteps), function(i) {
            paste0(local_bands, "_T", i)
          }))
          if (ncol(wide_mat) != length(wide_names)) {
            stop("Mismatch: number of columns is ", ncol(wide_mat), " but expected ", length(wide_names))
          }
          colnames(wide_mat) <- wide_names

          pixel_df <- as.data.frame(wide_mat)

          pred <- predict(local_model, newdata = pixel_df)

          pred_value <- as.numeric(pred)
          result_matrix <- matrix(rep(pred_value, local_nsteps), nrow = 1)

          return(result_matrix)
        }

      }

      prediction_cube <- tryCatch({
        gdalcubes::apply_time(
          data_cube,
          names = "prediction",
          keep_bands = FALSE,
          FUN = predict_time_fun
        )
      }, error = function(e) {
        message("Error during apply_time: ", conditionMessage(e))
        NULL
      })

      return(prediction_cube)
    }


    #######onnx prediction ################

    detected_model_type <- function(model) {
      if (endsWith(model, ".onnx")) {

        onnxruntime <- reticulate::import("onnxruntime")
        session <- onnxruntime$InferenceSession(model)
        input_details <- session$get_inputs()[[1]]

        if (length(input_details$shape) == 3) {
          Sys.setenv(ONNX_DL_FLAG = "TRUE")
          message("ONNX model detected as Deep Learning (3D input).")
        } else {
          Sys.setenv(ONNX_DL_FLAG = "FALSE")
          message("ONNX model detected as Classic ML (2D input).")
        }

        model <- list(
          session = session,
          input_name = input_details$name,
          input_shape = lapply(input_details$shape, function(x) if (is.null(x)) "None" else as.character(x)),
          is_onnx = TRUE
        )
        return(model)
      }
      stop("The transferred model is not an ONNX model.")
    }






    # Funktion für ONNX-Vorhersagen bei einem einzigen Zeitschritt (apply_pixel)
    mlm_single_onnx <- function(data_cube, model) {
      ensure_python_env(required_modules = c("numpy", "onnxruntime", "torch"))
      message("Preparing ONNX prediction (single time step) using apply_pixel()...")
      tmp <- Sys.getenv("SHARED_TEMP_DIR", tempdir())

      # Modellpfad – model muss ein String mit dem absoluten Pfad zum ONNX-Modell sein
      if (is.character(model)) {
        message("Model is a file path: ", model)
        model_file <- model
      } else {
        stop("Für ONNX-Vorhersagen muss model ein Pfad sein.")
      }
      Sys.setenv(MODEL_FILE = model_file)

      # Basis-Bandnamen speichern
      cube_band_names <- gdalcubes::bands(data_cube)$name
      band_names_file <- file.path(tmp, "band_names.rds")
      tryCatch({
        saveRDS(cube_band_names, band_names_file)
        message("Bandnamen-Datei gespeichert unter: ", normalizePath(band_names_file))
      }, error = function(e) {
        stop("Fehler beim Speichern der Bandnamen: ", e$message)
      })
      Sys.setenv(TMPDIRPATH = tmp)

      # Die Anzahl der Zeitschritte sollte 1 sein – überprüfen:
      time_steps <- gdalcubes::dimension_values(data_cube)$t
      nsteps <- length(time_steps)
      Sys.setenv(NSTEPS = nsteps)
      message("Anzahl Zeitschritte: ", nsteps)

      # Setze den Modelltyp (ONNX_DL_FLAG)
      model <- detected_model_type(model_file)
      message("Gehe jetzt in die predict_pixel_fun (ONNX)")

      # Callback-Funktion für apply_pixel()
      predict_pixel_fun <- function(x) {
        message("predict_pixel_fun (ONNX) gestartet")
        local_tmp <- Sys.getenv("TMPDIRPATH")
        nsteps <- as.numeric(Sys.getenv("NSTEPS"))
        model_file <- Sys.getenv("MODEL_FILE")

        if (!is.matrix(x)) {
          x <- matrix(x, nrow = 1)
        }

        local_bands <- readRDS(file.path(local_tmp, "band_names.rds"))
        if (is.null(colnames(x))) {
          colnames(x) <- local_bands
          message("Setze Spaltennamen: ", paste(local_bands, collapse = ", "))
        }

        onnx_dl_flag <- Sys.getenv("ONNX_DL_FLAG")
        if (onnx_dl_flag == "TRUE") {
          message("Deep Learning ONNX model detected; reshaping input to [1, n_channels, 1]")
          x <- array(x, dim = c(1, ncol(x), 1))
        } else {
          message("Classic ML ONNX model detected; using 2D input")
          x <- matrix(x, nrow = 1)
        }

        if (!reticulate::py_module_available("numpy")) {
          message("numpy nicht verfügbar – Installation wird versucht")
          reticulate::py_install("numpy", pip = TRUE)
        }
        if (!reticulate::py_module_available("onnxruntime")) {
          message("onnxruntime nicht verfügbar – Installation wird versucht")
          reticulate::py_install("onnxruntime", pip = TRUE)
          message("onnxruntime installiert")
        }
        np <- reticulate::import("numpy")
        onnxruntime <- reticulate::import("onnxruntime")

        session <- onnxruntime$InferenceSession(Sys.getenv("MODEL_FILE"))
        input_details <- session$get_inputs()[[1]]
        input_name <- input_details$name

        np_x <- reticulate::np_array(x, dtype = "float32")
        pred <- session$run(output_names = NULL,
                            input_feed = setNames(list(np_x), input_name))[[1]]
        if (onnx_dl_flag == "TRUE") {
          pred <- as.numeric(apply(pred, 1, which.max))
        }
        return(pred)
      }

      prediction_cube <- tryCatch({
        gdalcubes::apply_pixel(
          data_cube,
          names = "prediction",
          keep_bands = FALSE,
          FUN = predict_pixel_fun
        )
      }, error = function(e) {
        message("Error during apply_pixel: ", conditionMessage(e))
        NULL
      })

      return(prediction_cube)
    }

    mlm_multi_onnx <- function(data_cube, model) {
      ensure_python_env(required_modules = c("numpy", "onnxruntime", "torch"))
      message("Preparing ONNX prediction (multiple time steps) using apply_time()...")
      tmp <- Sys.getenv("SHARED_TEMP_DIR", tempdir())

      if (is.character(model)) {
        model_file <- model
      } else {
        stop("For ONNX predictions, model must be a path")
      }
      Sys.setenv(MODEL_FILE = model_file)

      cube_band_names <- gdalcubes::bands(data_cube)$name
      band_names_file <- file.path(tmp, "band_names.rds")
      saveRDS(cube_band_names, band_names_file)
      if (!file.exists(band_names_file)) {
        stop("The file band_names.rds was not saved successfully in : ", band_names_file)
      }
      Sys.setenv(TMPDIRPATH = tmp)

      time_steps <- gdalcubes::dimension_values(data_cube)$t
      nsteps <- length(time_steps)
      Sys.setenv(NSTEPS = nsteps)
      message("Number of time steps: ", nsteps)

      model <- detected_model_type(model_file)

      predict_time_fun <- function(x) {
        local_tmp <- Sys.getenv("TMPDIRPATH")
        nsteps_local <- as.numeric(Sys.getenv("NSTEPS"))
        model_file <- Sys.getenv("MODEL_FILE")

        if (!is.matrix(x)) {
          local_bands <- readRDS(file.path(local_tmp, "band_names.rds"))
          n_bands <- length(local_bands)
          x <- matrix(x, nrow = n_bands)
        }

        local_bands <- readRDS(file.path(local_tmp, "band_names.rds"))
        n_bands <- length(local_bands)
        if (is.null(colnames(x))) {
          colnames(x) <- local_bands
        }

        onnx_dl_flag <- Sys.getenv("ONNX_DL_FLAG")
        if (onnx_dl_flag == "TRUE") {
          x <- array(x, dim = c(1, n_bands, nsteps_local))
        } else {
          wide_vec <- as.vector(x)
          wide_mat <- matrix(wide_vec, nrow = 1, ncol = n_bands * nsteps_local)
          wide_names <- unlist(lapply(seq_len(nsteps_local), function(i) {
            paste0(local_bands, "_T", i)
          }))
          if (ncol(wide_mat) != length(wide_names)) {
            stop("Mismatch: number of columns is ", ncol(wide_mat), " but expected ", length(wide_names))
          }
          colnames(wide_mat) <- wide_names
          x <- wide_mat
        }

        if (!reticulate::py_module_available("numpy")) {
          reticulate::py_install("numpy", pip = TRUE)
        }
        if (!reticulate::py_module_available("onnxruntime")) {
          reticulate::py_install("onnxruntime", pip = TRUE)
        }
        np <- reticulate::import("numpy")
        onnxruntime <- reticulate::import("onnxruntime")

        session <- onnxruntime$InferenceSession(Sys.getenv("MODEL_FILE"))
        input_details <- session$get_inputs()[[1]]
        input_name <- input_details$name

        np_x <- reticulate::np_array(x, dtype = "float32")

        pred <- session$run(output_names = NULL,
                            input_feed = setNames(list(np_x), input_name))[[1]]
        if (onnx_dl_flag == "TRUE") {
          pred_class <- as.numeric(apply(pred, 1, which.max))
        } else {
          pred_class <- as.numeric(pred)
        }
        result <- matrix(rep(pred_class, nsteps_local), nrow = 1)
        message("Prediction: ", result)

        return(result)
      }

      prediction_cube <- tryCatch({
        gdalcubes::apply_time(
          data_cube,
          names = "prediction",
          keep_bands = FALSE,
          FUN = predict_time_fun
        )
      }, error = function(e) {
        message("Error during apply_time: ", conditionMessage(e))
        NULL
      })

      return(prediction_cube)
    }


    time_steps_query <- function(cube){
      time_steps <- gdalcubes::dimension_values(cube)
      time_steps <- time_steps$t
      return(time_steps)
    }

    #######################################
    time <- time_steps_query(cube)
    message("ml_pedict starting...")
    if (is.raw(model)) {
      message("RAW model recognized - try to determine type")
      model <- tryCatch({
        model_obj <- readRDS(tmp_file)
        message("RAW was a .rds model - model loaded.")
        con_rds <- rawConnection(model_obj, "rb")
        model <- torch::torch_load(con_rds)
      }, error = function(e) {
        tmp_file <- tempfile(fileext = ".onnx")
        writeBin(model, tmp_file)
        message("No .rds recognized - treat as .onnx: ", tmp_file)
        return(tmp_file)
      })
    }
    if (is.list(model) && !is.null(model$onnx) && endsWith(model$onnx, ".onnx")) {
      message("Model provided as list – using ONNX: ", model$onnx)
      model <- model$onnx
    }

    #time <- time_steps_query(cube)
    band_info <- gdalcubes::bands(cube)
    band_names <- band_info$name
    cube_dimensions <- gdalcubes::dimensions(cube)
    time_count <- cube_dimensions$t$count
    multi_timesteps <- time_count > 1
    input_channels <- length(band_names)

    if (is.character(model) && endsWith(model, ".onnx")) {
      if (multi_timesteps) {
        message("ONNX model detected – multi time steps")
        return(mlm_multi_onnx(cube, model))
      } else {
        message("ONNX model detected – single time step")
        return(mlm_single_onnx(cube, model))
      }
    }

    if (is.character(model)) {
      message("Loading external model...")

      if (endsWith(model, ".pt")) {
        library(torch)

        if (!file.exists(model)) {
          stop("Model file does not exist: ", model)
        }
        model <- torch::torch_load(model)
        message("Torch model successfully loaded.")

      } else if (endsWith(model, ".rds")) {
        model_obj <- readRDS(model)
        message("RDS model loaded successfully.")

        if(is.raw(model_obj)){
          message("raw RDS detetcted")
          con_rds <- rawConnection(model_obj, "rb")
          model <- torch::torch_load(con_rds)
        }
        else if (inherits(model_obj, "train")) {
          message("Caret model recognized")
          model <- model_obj
        } else {
          stop("Unknown model type in .rds - no nn_module, no caret model, no Torch state_dict")
        }
      }else {
        stop("Unsupported model format")
      }
    }
    if (multi_timesteps) {
      message("prediction for multi time steps")
      prediction <- mlm_multi(cube, model)
      message("prediction successful")
      return(prediction)
    } else {
      message("Only one time step")
      prediction <- mlm_single(cube, model)
      return(prediction)
    }
  }
)

########################################################
##### mlm_svm_process #####
#' Support Vector Machine for classification or regression
#'
#' @description
#' Model case for support vector machines. Creates a parameter list and trainControl
#' settings for SVM models (classification or regression) with specified kernel and hyperparameters.
#'
#' @param kernel Character. Kernel type to use: `'radial'`, `'linear'`, or `'polynomial'`.
#' @param C Numeric. Regularization parameter (cost).
#' @param sigma Numeric, optional. Kernel coefficient (σ) for the radial basis function kernel.
#' @param gamma Numeric, optional. Kernel coefficient (γ) for the polynomial kernel (also scales the polynomial).
#' @param degree Integer, optional. Degree of the polynomial kernel (default is 3).
#' @param coef0 Numeric, optional. Independent term in polynomial kernel (default is 0).
#' @param random_state Integer, optional. Random seed for reproducibility.
#' @param classification Logical. If `TRUE`, sets up SVM for classification; if `FALSE`, for regression.
#'
#' @return
#' A list of model parameters, including:
#' - `method`: the caret model string (`"svmRadial"`, `"svmLinear"`, or `"svmPoly"`)
#' - `tuneGrid`: a data.frame with the tuning grid
#' - `trControl`: a `trainControl` object with 5-fold cross-validation
#' - `seed`: the random seed (if provided)
#'
#' @examples
#' # Classification with radial kernel
#' params <- mlm_svm_process(kernel = "radial", C = 1, sigma = 0.1, classification = TRUE)
#'
#' # Regression with polynomial kernel
#' params <- mlm_svm_process(kernel = "polynomial", C = 0.5, degree = 4, gamma = 0.01, coef0 = 1, classification = FALSE)
#'

mlm_svm_process <- Process$new(
  id = "mlm_svm",
  description = "Model case for support vector machines.",
  categories = c("machine-learning"),
  summary = "Support Vector Machine for classification or regression",

  parameters = list(
    Parameter$new(
      name = "kernel",
      description = "Kernel type ('radial', 'linear', 'polynomial')",
      schema = list(type = "string")
    ),
    Parameter$new(
      name = "C",
      description = "Regularization parameter",
      schema = list(type = "number")
    ),
    Parameter$new(
      name = "sigma",
      description = "Kernel coefficient for radial basis function",
      schema = list(type = "number"),
      optional = TRUE
    ),
    Parameter$new(
      name = "gamma",
      description = "Kernel coefficient for polynomial kernel",
      schema = list(type = "number"),
      optional = TRUE
    ),
    Parameter$new(
      name = "degree",
      description = "Degree for polynomial kernel",
      schema = list(type = "integer"),
      optional = TRUE
    ),
    Parameter$new(
      name = "coef0",
      description = "Independent term in polynomial kernel",
      schema = list(type = "number"),
      optional = TRUE
    ),
    Parameter$new(
      name = "random_state",
      description = "Random seed for reproducibility",
      schema = list(type = "integer"),
      optional = TRUE
    ),
    Parameter$new(
      name = "classification",
      description = "TRUE for classification; FALSE for regression",
      schema = list(type = "boolean")
    )
  ),

  returns = list(
    description = "Model parameters as a list for the SVM model",
    schema = list(
      type = "object",
      subtype = "mlm-model"
    )
  ),

  operation = function(kernel,
                       C,
                       sigma = NULL,
                       gamma = NULL,
                       degree = 3,
                       coef0 = 0,
                       random_state = NULL,
                       classification,
                       job) {
    message("SVM envelope is created")

    mlm_class_svm <- function(kernel, C, sigma, gamma, degree, coef0, random_state) {
      if (kernel == "radial") {
        tuneGrid <- expand.grid(C = C, sigma = sigma)
        method <- "svmRadial"
      } else if (kernel == "linear") {
        tuneGrid <- expand.grid(C = C)
        method <- "svmLinear"
      } else if (kernel == "polynomial") {
        tuneGrid <- expand.grid(C = C, degree = degree, scale = gamma, coef0 = coef0)
        method <- "svmPoly"
      } else {
        stop("Unsupported kernel type.")
      }

      return(list(
        method = method,
        tuneGrid = tuneGrid,
        trControl = caret::trainControl(
          method = "cv",
          number = 5
        ),
        seed = random_state
      ))
    }

    mlm_regr_svm <- function(kernel, C, sigma, gamma, degree, coef0, random_state) {
      if (kernel == "radial") {
        tuneGrid <- expand.grid(C = C, sigma = sigma)
        method <- "svmRadial"
      } else if (kernel == "linear") {
        tuneGrid <- expand.grid(C = C)
        method <- "svmLinear"
      } else if (kernel == "polynomial") {
        tuneGrid <- expand.grid(C = C, degree = degree, scale = gamma, coef0 = coef0)
        method <- "svmPoly"
      } else {
        stop("Unsupported kernel type.")
      }

      return(list(
        method = method,
        tuneGrid = tuneGrid,
        trControl = caret::trainControl(
          method = "cv",
          number = 5
        ),
        seed = random_state
      ))
    }

    if (classification) {
      message("SVM envelope successfully created (classification)")
      return(mlm_class_svm(kernel, C, sigma, gamma, degree, coef0, random_state))
    } else {
      message("SVM envelope successfully created (regression)")
      return(mlm_regr_svm(kernel, C, sigma, gamma, degree, coef0, random_state))
    }
  }
)

########################################################
#' Random forest for classification or regression
#'
#' @description
#' Model case for the random forest. Creates a parameter list and trainControl
#' settings for Random Forest models (classification or regression) with specified
#' number of trees and splitting criteria.
#'
#' @param num_trees Integer. Number of trees in the forest.
#' @param min_samples_split Integer. Minimum number of observations required to split an internal node.
#' @param min_samples_leaf Integer. Minimum number of observations required to be at a leaf node.
#' @param max_features String. Number of features to consider when looking for the best split (e.g., `"sqrt"` or a numeric value).
#' @param random_state Integer, optional. Random seed for reproducibility.
#' @param classification Logical. If `TRUE`, sets up Random Forest for classification; if `FALSE`, for regression.
#'
#' @return
#' A list of model parameters for caret, including:
#' - `method`: `"rf"`
#' - `tuneGrid`: a `data.frame` with the `mtry` value
#' - `trControl`: a `trainControl` object with 5-fold cross-validation
#' - `ntree`: number of trees
#' - `seed`: the random seed (if provided)
#' - `min_samples_split` and `min_samples_leaf`: node splitting criteria
#'
#' @examples
#' # Classification with 100 trees, sqrt(max_features)
#' params <- mlm_random_forest(num_trees = 100,
#'                             min_samples_split = 2,
#'                             min_samples_leaf = 1,
#'                             max_features = "sqrt",
#'                             classification = TRUE)
#'
#' # Regression with 200 trees, 5 features per split
#' params <- mlm_random_forest(num_trees = 200,
#'                             min_samples_split = 5,
#'                             min_samples_leaf = 2,
#'                             max_features = "5",
#'                             random_state = 42,
#'                             classification = FALSE)
#'
mlm_random_forest <- Process$new(
  id = "mlm_random_forest",
  description = "Model case for the random forest .",
  categories = c("machine-learning"),
  summary = "Random forest for classification or regression",

  parameters = list(
    Parameter$new(
      name = "num_trees",
      description = "number of trees",
      schema = list(type = "integer")
    ),
    Parameter$new(
      name = "min_samples_split",
      description = "Minimum number of observations to divide a node",
      schema = list(type = "integer")
    ),
    Parameter$new(
      name = "min_samples_leaf",
      description = "Minimum number of observations that must be present in a sheet",
      schema = list(type = "integer")
    ),
    Parameter$new(
      name = "max_features",
      description = "Number of characteristics to be considered for the best split (e.g. 'sqrt' or a number)",
      schema = list(type = "string")
    ),
    Parameter$new(
      name = "random_state",
      description = "Random seed for reproducibility",
      schema = list(type = "integer"),
      optional = TRUE
    ),
    Parameter$new(
      name = "classification",
      description = "TRUE, falls Klassifikation; FALSE für Regression",
      schema = list(type = "boolean")
    )
  ),

  returns = list(
    description = "Model parameters as a list for the Random Forest model",
    schema = list(
      type = "object",
      subtype = "mlm-model"
    )
  ),

  operation = function(num_trees,
                       min_samples_split,
                       min_samples_leaf,
                       max_features,
                       random_state = NULL,
                       classification,
                       job) {
    message("Cover for the random forest is created")

    mlm_class_random_forest <- function(num_trees, min_samples_split, min_samples_leaf, max_features, random_state) {
      model_params <- list(
        method = "rf",
        tuneGrid = expand.grid(mtry = if (max_features == "sqrt") NA else max_features),
        trControl = caret::trainControl(
          method = "cv",
          number = 5
        ),
        ntree = num_trees,
        seed = random_state,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf
      )
      return(model_params)
    }

    mlm_regr_random_forest <- function(num_trees, min_samples_split, min_samples_leaf, max_features, random_state) {
      model_params <- list(
        method = "rf",
        tuneGrid = expand.grid(mtry = if (max_features == "sqrt") NA else max_features),
        trControl = caret::trainControl(
          method = "cv",
          number = 5
        ),
        ntree = num_trees,
        seed = random_state,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf
      )
      return(model_params)
    }

    if (classification) {
      message("RF envelope successfully created (classification)")
      return(mlm_class_random_forest(num_trees, min_samples_split, min_samples_leaf, max_features, random_state))
    } else {
      message("RF envelope successfully created (regression)")
      return(mlm_regr_random_forest(num_trees, min_samples_split, min_samples_leaf, max_features, random_state))
    }
  }
)
########################################################
#' Extreme Gradient Boosting for classification or regression
#'
#' @description
#' Model case for XGBoost. Creates a parameter list and trainControl settings
#' for XGBoost models (classification or regression) with specified hyperparameters.
#'
#' @param learning_rate Numeric. Step size shrinkage (eta) to prevent overfitting.
#' @param max_depth Integer. Maximum depth of each tree.
#' @param min_child_weight Numeric. Minimum sum of instance weight (hessian) needed in a child.
#' @param subsample Numeric. Subsample ratio of the training instances.
#' @param colsample_bytree Numeric. Subsample ratio of columns when constructing each tree.
#' @param gamma Numeric. Minimum loss reduction required to make a further partition.
#' @param nrounds Integer. Number of boosting iterations.
#' @param random_state Integer, optional. Random seed for reproducibility.
#' @param classification Logical. If `TRUE`, sets up XGBoost for classification; if `FALSE`, for regression.
#'
#' @return
#' A list of model parameters for caret's `xgbTree` method, including:
#' - `method`: `"xgbTree"`
#' - `tuneGrid`: a `data.frame` of hyperparameters (`nrounds`, `max_depth`, `eta`, `gamma`, `colsample_bytree`, `min_child_weight`, `subsample`)
#' - `trControl`: a `trainControl` object with 5-fold cross-validation and grid search
#' - `random_state`: the random seed (if provided)
#'
#' @examples
#' # Classification example
#' params <- mlm_xgboost_process(
#'   learning_rate    = 0.1,
#'   max_depth        = 6,
#'   min_child_weight = 1,
#'   subsample        = 0.8,
#'   colsample_bytree = 0.8,
#'   gamma            = 0,
#'   nrounds          = 100,
#'   classification   = TRUE
#' )
#'
#' # Regression example
#' params <- mlm_xgboost_process(
#'   learning_rate    = 0.05,
#'   max_depth        = 4,
#'   min_child_weight = 3,
#'   subsample        = 0.7,
#'   colsample_bytree = 0.7,
#'   gamma            = 1,
#'   nrounds          = 200,
#'   random_state     = 42,
#'   classification   = FALSE
#' )
mlm_xgboost_process <- Process$new(
  id = "mlm_xgboost",
  description = "Model case for XGBoost.",
  categories = c("machine-learning"),
  summary = "Extreme Gradient Boosting for classification or regression",

  parameters = list(
    Parameter$new(
      name = "learning_rate",
      description = "Step size shrinkage used in update to prevents overfitting",
      schema = list(type = "number")
    ),
    Parameter$new(
      name = "max_depth",
      description = "Maximum depth of a tree",
      schema = list(type = "integer")
    ),
    Parameter$new(
      name = "min_child_weight",
      description = "Minimum sum of instance weight (hessian) needed in a child",
      schema = list(type = "number")
    ),
    Parameter$new(
      name = "subsample",
      description = "Subsample ratio of the training instances",
      schema = list(type = "number")
    ),
    Parameter$new(
      name = "colsample_bytree",
      description = "Subsample ratio of columns when constructing each tree",
      schema = list(type = "number")
    ),
    Parameter$new(
      name = "gamma",
      description = "Minimum loss reduction required to make a further partition",
      schema = list(type = "number")
    ),
    Parameter$new(
      name = "nrounds",
      description = "Number of boosting iterations",
      schema = list(type = "integer")
    ),
    Parameter$new(
      name = "random_state",
      description = "Random seed for reproducibility",
      schema = list(type = "integer"),
      optional = TRUE
    ),
    Parameter$new(
      name = "classification",
      description = "TRUE for classification; FALSE for regression",
      schema = list(type = "boolean")
    )
  ),

  returns = list(
    description = "Model parameters as a list for the XGBoost model",
    schema = list(
      type = "object",
      subtype = "mlm-model"
    )
  ),

  operation = function(learning_rate,
                       max_depth,
                       min_child_weight,
                       subsample,
                       colsample_bytree,
                       gamma,
                       nrounds,
                       random_state = NULL,
                       classification,
                       job) {
    message("XGBoost envelope is created")

    mlm_class_xgboost <- function(learning_rate, max_depth, min_child_weight,
                                  subsample, colsample_bytree, gamma, nrounds, random_state) {
      return(list(
        method = "xgbTree",
        tuneGrid = expand.grid(
          nrounds = nrounds,
          max_depth = max_depth,
          eta = learning_rate,
          gamma = gamma,
          colsample_bytree = colsample_bytree,
          min_child_weight = min_child_weight,
          subsample = subsample
        ),
        trControl = caret::trainControl(
          method = "cv",
          number = 5,
          search = "grid"
        ),
        random_state = random_state
      ))
    }

    mlm_regr_xgboost <- function(learning_rate, max_depth, min_child_weight,
                                 subsample, colsample_bytree, gamma, nrounds, random_state) {
      return(list(
        method = "xgbTree",
        tuneGrid = expand.grid(
          nrounds = nrounds,
          max_depth = max_depth,
          eta = learning_rate,
          gamma = gamma,
          colsample_bytree = colsample_bytree,
          min_child_weight = min_child_weight,
          subsample = subsample
        ),
        trControl = caret::trainControl(
          method = "cv",
          number = 5,
          search = "grid"
        ),
        random_state = random_state
      ))
    }

    if (classification) {
      message("XGBoost envelope successfully created (classification)")
      return(mlm_class_xgboost(learning_rate, max_depth, min_child_weight,
                               subsample, colsample_bytree, gamma, nrounds, random_state))
    } else {
      message("XGBoost envelope successfully created (regression)")
      return(mlm_regr_xgboost(learning_rate, max_depth, min_child_weight,
                              subsample, colsample_bytree, gamma, nrounds, random_state))
    }
  }
)

####################################################
#' TempCNN for classification or regression
#'
#' @description
#' Model case for the TempCNN model. Defines convolutional and dense layer settings,
#' optimizer parameters, and returns a factory function to create the TempCNN module.
#'
#' @param cnn_layer Integer vector. Number of filters in each convolutional layer.
#' @param cnn_kernels Integer vector. Kernel sizes for each convolutional layer.
#' @param cnn_dropout_rates Numeric vector. Dropout rates for each convolutional layer.
#' @param dense_layer_nodes Integer. Number of neurons in the dense (fully-connected) layer.
#' @param dense_layer_dropout_rate Numeric. Dropout rate for the dense layer.
#' @param optimizer Character. Name of the optimizer (e.g., `"adam"`).
#' @param learning_rate Numeric. Learning rate for the optimizer.
#' @param epsilon Numeric. Epsilon value for numerical stability in the optimizer.
#' @param weight_decay Numeric. Weight decay (L2 regularization) coefficient.
#' @param lr_decay_epochs Integer. Number of epochs after which to decay the learning rate.
#' @param lr_decay_rate Numeric. Factor by which to decay the learning rate.
#' @param epochs Integer. Total number of training epochs.
#' @param batch_size Integer. Batch size for training.
#' @param random_state Integer, optional. Random seed for reproducibility.
#'
#' @return
#' A list with two elements:
#' \describe{
#'   \item{parameters}{A named list of all TempCNN hyperparameters.}
#'   \item{create_model}{A function that, given `input_data_columns`, `time_steps`, and `class_count`, returns an initialized TempCNN `nn_module`.}
#' }
mlm_class_tempcnn <- Process$new(
  id = "mlm_class_tempcnn",
  description = "Model case for the TempCNN model.",
  categories = as.array("machine-learning"),
  summary = "TempCNN for classification or regression",
  parameters = list(
    Parameter$new(
      name = "cnn_layer",
      description = "Array: Number of filters per convolutional layer",
      schema = list(type = "array", items = list(type = "integer"))
    ),
    Parameter$new(
      name = "cnn_kernels",
      description = "Array: Kernel sizes for the convolutional layers",
      schema = list(type = "array", items = list(type = "integer"))
    ),
    Parameter$new(
      name = "cnn_dropout_rates",
      description = "Array: Dropout rates for the convolutional layers",
      schema = list(type = "array", items = list(type = "number"))
    ),
    Parameter$new(
      name = "dense_layer_nodes",
      description = "Number of neurons in the dense layer",
      schema = list(type = "integer")
    ),
    Parameter$new(
      name = "dense_layer_dropout_rate",
      description = "Dropout rate for the dense layer",
      schema = list(type = "number")
    ),
    Parameter$new(
      name = "optimizer",
      description = "Optimizer name (e.g. 'adam')",
      schema = list(type = "string")
    ),
    Parameter$new(
      name = "learning_rate",
      description = "Learning rate for the optimizer",
      schema = list(type = "number")
    ),
    Parameter$new(
      name = "epsilon",
      description = "Epsilon value for the optimizer",
      schema = list(type = "number")
    ),
    Parameter$new(
      name = "weight_decay",
      description = "Weight loss (regularization)",
      schema = list(type = "number")
    ),
    Parameter$new(
      name = "lr_decay_epochs",
      description = "Number of epochs after which the learning rate decreases",
      schema = list(type = "integer")
    ),
    Parameter$new(
      name = "lr_decay_rate",
      description = "Rate of learning rate decay",
      schema = list(type = "number")
    ),
    Parameter$new(
      name = "epochs",
      description = "Number of training epochs",
      schema = list(type = "integer")
    ),
    Parameter$new(
      name = "batch_size",
      description = "Batch size for training",
      schema = list(type = "integer")
    ),
    Parameter$new(
      name = "random_state",
      description = "Random generator for reproducibility",
      schema = list(type = "integer"),
      optional = TRUE
    )
  ),
  returns = list(
    description = "Model parameters as a list for the TempCNN model",
    schema = list(type = "object", subtype = "mlm-model")
  ),
  operation = function(cnn_layer = cnn_layer,
                       cnn_kernels = cnn_kernels,
                       cnn_dropout_rates = cnn_dropout_rates,
                       dense_layer_nodes = dense_layer_nodes,
                       dense_layer_dropout_rate = dense_layer_dropout_rate,
                       optimizer = optimizer,
                       learning_rate = learning_rate,
                       epsilon = epsilon,
                       weight_decay = weight_decay,
                       lr_decay_epochs = lr_decay_epochs,
                       lr_decay_rate = lr_decay_rate,
                       epochs = epochs,
                       batch_size = batch_size,
                       random_state = random_state, job) {


    model_parameter <- list(
      cnn_layer = cnn_layer,
      cnn_kernels = cnn_kernels,
      cnn_dropout_rates = cnn_dropout_rates,
      dense_layer_nodes = dense_layer_nodes,
      dense_layer_dropout_rate = dense_layer_dropout_rate,
      optimizer = optimizer,
      learning_rate = learning_rate,
      epsilon = epsilon,
      weight_decay = weight_decay,
      lr_decay_epochs = lr_decay_epochs,
      lr_decay_rate = lr_decay_rate,
      epochs = epochs,
      batch_size = batch_size,
      random_state = random_state
    )

    return(list(
      parameters = model_parameter,
      create_model = function(input_data_columns, time_steps, class_count) {
        dl_temp_cnn_dynamic <- nn_module(
          initialize = function(settings, input_data_columns, time_steps, class_count) {
            self$conv_layers <- nn_module_list()
            self$input_data_columns <- input_data_columns
            self$original_time_steps <- time_steps

            reduced_time_steps <- time_steps
            for (i in seq_along(settings$cnn_layer)) {
              input_channels <- ifelse(
                i == 1,
                length(self$input_data_columns),
                settings$cnn_layer[[i - 1]]
              )

              output_channels <- settings$cnn_layer[[i]]
              kernel_size <- settings$cnn_kernels[[i]]
              dropout_rate <- settings$cnn_dropout_rates[[i]]

              self$conv_layers$append(
                nn_sequential(
                  nn_conv1d(
                    in_channels = input_channels,
                    out_channels = output_channels,
                    kernel_size = kernel_size,
                    stride = 1,
                    padding = kernel_size %/% 2
                  ),
                  nn_batch_norm1d(output_channels),
                  nn_relu(),
                  nn_dropout(p = dropout_rate)
                )
              )

              reduced_time_steps <- (reduced_time_steps + 2 * (kernel_size %/% 2) - kernel_size) + 1
            }

            self$time_steps <- reduced_time_steps
            self$flatten <- nn_flatten()
            self$dense <- nn_sequential(
              nn_linear(settings$cnn_layer[[length(settings$cnn_layer)]] * self$time_steps, settings$dense_layer_nodes),
              nn_relu(),
              nn_dropout(p = settings$dense_layer_dropout_rate),
              nn_linear(settings$dense_layer_nodes, class_count)
            )
          },

          forward = function(x) {
            for (i in seq_along(self$conv_layers)) {
              x <- self$conv_layers[[i]](x)
            }
            x <- self$flatten(x)
            x <- self$dense(x)
            return(x)
          }
        )
        return(dl_temp_cnn_dynamic(
          settings = model_parameter,
          input_data_columns = input_data_columns,
          time_steps = time_steps,
          class_count = class_count
        ))
      }
    ))
  }
)
#######################################
#' Load an ML model from a URL (asynchronous subprocess)
#'
#' @description
#' Downloads a machine learning model file (`.rds` or `.onnx`) from the specified URL.
#' Supports internal URLs (e.g. `http://localhost:8000/download/…`) by loading directly
#' from the shared temp directory, and external URLs via HTTP(S). Google Drive links
#' are auto-converted to direct download URLs.
#'
#' @param url Character. URL from which to load the model file.
#'
#' @return
#' If the model is `.rds`, returns the loaded R object; if `.onnx`, returns a raw vector
#' of the ONNX file contents.
#'
load_ml_model <- Process$new(
  id = "load_ml_model",
  description = "Loads a machine learning model (.rds or .onnx) directly from a URL - asynchronously via subprocess.",
  categories = as.array("model-management", "data-loading"),
  summary = "Loads a model from a URL (only .rds or .onnx), even if it comes from the same server.",

  parameters = list(
    Parameter$new(
      name = "url",
      description = "The URL from which the model is to be loaded",
      schema = list(type = "string")
    )
  ),

  returns = ml_model,

  operation = function(url, job) {
    library(tools)
    library(httr)
    message("load_ml_model is started...")

    message("Receive model URL: ", url)
    if (grepl("^http://localhost:8000/download/", url)) {
      message("Internal URL recognized - file path is used directly")
      file_name <- basename(url)
      shared_dir <- Sys.getenv("SHARED_TEMP_DIR", tempdir())
      file_path <- file.path(shared_dir, file_name)

      if (!startsWith(normalizePath(file_path), normalizePath(shared_dir))) {
        stop("Access outside the permitted directory!")
      }
      if (!file.exists(file_path)) {
        stop("File was not found: ", file_path)
      }
      ext <- tolower(file_ext(file_path))
      if (ext == "rds") {
        return(file_path)
      } else if (ext == "onnx") {
        return(readBin(file_path, "raw", n = file.info(file_path)$size))
      } else {
        stop("Only .rds and .onnx are supported (currently: ", ext, ")")
      }
    }
    message("External URL recognized - try HTTP access")
    if (grepl("drive\\.google\\.com/file/d/", url)) {
      drive_match <- regmatches(url, regexec("drive\\.google\\.com/file/d/([^/?]+)", url))[[1]]
      if (length(drive_match) > 1) {
        file_id <- drive_match[2]
        url <- paste0("https://drive.google.com/uc?export=download&id=", file_id)
        message("Google Drive link recognized and converted: ", url)
      } else {
        stop("Could not extract a valid Google Drive file ID.")
      }
    }
    message("response is provided")
    response <- httr::GET(url)
    if (httr::status_code(response) != 200) {
      stop("Error retrieving the URL: ", httr::status_code(response))
    }
    content_disposition <- headers(response)[["content-disposition"]]
    file_name <- basename(url)
    if (!is.null(content_disposition)) {
      match <- regmatches(content_disposition, regexec('filename="?([^";]+)"?', content_disposition))[[1]]
      if (length(match) > 1) {
        file_name <- match[2]
      }
    }
    ext <- tolower(file_ext(file_name))
    raw_data <- content(response, "raw")
    if (ext == "rds") {
      tmp <- tempfile(fileext = ".rds")
      writeBin(raw_data, tmp)
      model <- readRDS(tmp)
      return(model)

    } else if (ext == "onnx") {
      return(raw_data)

    } else {
      stop("Only .rds and .onnx are supported (currently: ", ext, ")")
    }
  }
)
#######################################################
#' Save trained ML model as ONNX, JSON (MLM-STAC), and RDS
#'
#' @description
#' Saves a trained machine learning model in ONNX, JSON (MLM-STAC) and RDS formats.
#' For Torch (nn_module) models, also produces a TorchScript (.pt) version.
#' Handles caret `train` objects and torch `nn_module`s, enriches ONNX with metadata,
#' and returns download links for all artifacts.
#'
#' @param data Object of subtype "mlm-model". A trained ML model: a caret `train` object or a torch `nn_module`.
#' @param name Character. Unique identifier for the model; used as base filename for saved artifacts.
#' @param tasks List. The use case(s) of the model (e.g., `list("classification")` or `list("regression")`).
#' @param options List, optional. Additional key-value options for metadata (e.g., `list("mlm:pretrained" = TRUE)`).
#'
#' @return
#' A named list with file paths to the exported artifacts:
#' \describe{
#'   \item{onnx}{Path to the saved ONNX model file}
#'   \item{rds}{Path to the saved RDS (or raw RDS) file}
#'   \item{json}{Path to the saved MLM-STAC JSON metadata file}
#' }
save_ml_model <- Process$new(
  id = "save_ml_model",
  description = "Saves a trained machine learning model in ONNX, JSON (MLM-STAC) and RDS formats. For Torch models, a TorchScript version is also produced.",
  categories = as.array("cubes", "Machine Learning"),
  summary = "Save trained ML model as ONNX, JSON, and RDS",
  parameters = list(
    Parameter$new(
      name = "data",
      description = "A trained ML model.",
      schema = list(type = "object", subtype = "mlm-model")
    ),
    Parameter$new(
      name = "name",
      description = "The unique identifier for the model.",
      schema = list(type = "string")
    ),
    Parameter$new(
      name = "tasks",
      description = "The use case(s) of the model (e.g., classification or regression).",
      schema = list(type = "list")
    ),
    Parameter$new(
      name = "options",
      description = "Additional options as key-value pairs.",
      schema = list(type = "object"),
      optional = TRUE
    )
  ),
  returns = ml_model,
  operation = function(data, name, tasks, options, job) {
    ensure_python_env(required_modules = c("numpy", "onnxruntime", "torch", "joblib", "scikit-learn", "xgboost", "onnxmltools", "skl2onnx"))
    message("✅ ensure_python_env wurde aufgerufen")

    message("Save model is started...")
    shared_dir <- Sys.getenv("SHARED_TEMP_DIR", tempdir())


    ensure_extension <- function(filename, ext) {
      if (!grepl(paste0("\\.", ext, "$"), filename, ignore.case = TRUE)) {
        return(paste0(filename, ".", ext))
      }
      return(filename)
    }
    detect_model_type <- function(model) {
      if ("train" %in% class(model)) {
        label <- model$modelInfo$label
        if (grepl("Random Forest", label, ignore.case = TRUE)) {
          return("random_forest")
        } else if (grepl("Support Vector Machines", label, ignore.case = TRUE)) {
          if (grepl("Linear", label, ignore.case = TRUE)) {
            return("svmLinear")
          } else if (grepl("Radial", label, ignore.case = TRUE)) {
            return("svmRadial")
          } else if (grepl("Poly", label, ignore.case = TRUE)) {
            return("svmPoly")
          } else {
            return("svmLinear")
          }
        } else if (grepl("xgboost", label, ignore.case = TRUE) ||
                   grepl("Gradient Boosting", label, ignore.case = TRUE) ||
                   grepl("Xtreme", label, ignore.case = TRUE)) {
          return("xgbTree")
        } else {
          stop("Model type could not be recognized from model$modelInfo$label: ", label)
        }
      } else {
        return(NULL)
      }
    }

    save_torch_model <- function(model, filepath) {
      if (!is.null(model$conv_layers[[1]])) {
        first_conv_layer <- model$conv_layers[[1]][[1]]
        input_channels <- first_conv_layer$in_channels
      } else {
        stop("Error: Could not recognize input_channels")
      }
      time_steps <- model$time_steps
      dummy_input <- torch::torch_randn(c(1, input_channels, time_steps))
      script_model <- tryCatch({
        torch::jit_trace(model, dummy_input)
      }, error = function(e) {
        torch::jit_script(model)
      })
      torch::jit_save(script_model, filepath)
      return(list(filepath = filepath, channels = input_channels, time_steps = time_steps))
    }
    convert_r_torch_to_onnx <- function(model, name) {
      if (!reticulate::py_module_available("torch")) {
        reticulate::py_install("torch", pip = TRUE)
      }
      model_info <- save_torch_model(model, paste0(name, ".pt"))
      filepath_script <- model_info$filepath
      channels <- model_info$channels
      time_steps <- model_info$time_steps

      output_file <- ensure_extension(name, "onnx")

      python_code <- sprintf('
import torch
import torch.onnx
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx.utils")

model = torch.jit.load("%s")
model.eval()

dummy_input = torch.randn(1, %d, %d)
torch.onnx.export(model, dummy_input, "%s",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={
                      "input": {0: "batch_size", 2: "time_steps"},
                      "output": {0: "batch_size"}
                  })
print("ONNX model successfully saved: %s")
', filepath_script, channels, time_steps, output_file, output_file)

      py_file <- tempfile(fileext = ".py")
      writeLines(python_code, py_file)
      command <- sprintf('python3 %s', shQuote(py_file))
      result <- system(command, intern = TRUE)
      unlink(py_file)

      message(result)
      return(output_file)
    }

    convert_model_to_pkl <- function(model, model_type, filepath) {
      library(reticulate)
      joblib <- import("joblib")
      np <- import("numpy")
      sklearn <- import("sklearn.ensemble")
      sklearn_svm <- import("sklearn.svm")
      xgboost <- import("xgboost")

      if (!("train" %in% class(model))) {
        stop("Please pass a caret train object")
      }

      if (!("trainingData" %in% names(model))) {
        stop("The caret model does not contain any trainingData. Please save the model with trainingData")
      }
      train_data <- model$trainingData

      if (!(".outcome" %in% colnames(train_data))) {
        stop("The trainingData does not contain an '.outcome' column")
      }
      predictors <- setdiff(colnames(train_data), ".outcome")
      target_column <- ".outcome"
      model <- model$finalModel
      train_data_clean <- train_data[, predictors, drop = FALSE]
      if (is.factor(train_data[[target_column]]) || is.character(train_data[[target_column]])) {
        train_data[[target_column]] <- as.integer(as.factor(train_data[[target_column]])) - 1
      }
      x_train <- np$array(as.matrix(train_data_clean))
      y_train <- np$array(as.numeric(train_data[[target_column]]))

      if (model_type == "random_forest") {
        if (!inherits(model, "randomForest")) {
          stop("Error: The model is not a RandomForest model")
        }
        rf_py_model <- sklearn$RandomForestClassifier(
          n_estimators = as.integer(model$ntree),
          max_features = as.integer(model$mtry)
        )
        rf_py_model$fit(x_train, y_train)
        joblib$dump(rf_py_model, paste0(filepath, ".pkl"))
      } else if (model_type %in% c("svmLinear", "svmRadial", "svmPoly")) {
        if (!inherits(model, "ksvm") && !("svm" %in% class(model))) {
          stop("Error: The model is not an SVM model")
        }
        C_value <- if (!is.null(model@kernelf@kpar$C)) model@kernelf@kpar$C else 1.0
        kernel <- if (model_type == "svmLinear") "linear" else "rbf"
        svm_py_model <- sklearn_svm$SVC(
          kernel = kernel,
          C = as.numeric(C_value)
        )
        svm_py_model$fit(x_train, y_train)
        joblib$dump(svm_py_model, paste0(filepath, ".pkl"))
      } else if (model_type == "xgbTree") {
        if (!inherits(model, "xgb.Booster")) {
          stop("Error: The model is not an XGBoost model")
        }
        xgboost::xgb.save(model, paste0(filepath, ".bin"))
      } else {
        stop("Model type is not supported!")
      }
    }

    add_metadata_to_onnx <- function(onnx_path, options) {
      onnx <- reticulate::import("onnx")
      model <- onnx$load_model(onnx_path)
      for (key in names(options)) {
        value <- as.character(options[[key]])
        meta_prop <- onnx$StringStringEntryProto(key = key, value = value)
        model$metadata_props$append(meta_prop)
      }
      onnx$save_model(model, onnx_path)
      message("Metadata has been added to the ONNX model.")
    }

    save_ml_model_as_onnx <- function(model_type, filepath) {
      library(reticulate)
      onnxmltools <- reticulate::import("onnxmltools")
      skl2onnx <- reticulate::import("skl2onnx")
      xgboost <- reticulate::import("xgboost")
      sklearn <- reticulate::import("sklearn.ensemble")
      sklearn_svm <- reticulate::import("sklearn.svm")
      onnx <- reticulate::import("onnx")
      joblib <- reticulate::import("joblib")
      if (model_type == "xgbTree") {
        xgb_model_py <- xgboost$XGBClassifier()
        xgb_model_py$load_model(paste0(filepath, ".bin"))
        n_features <- xgb_model_py$n_features_in_
      } else if (model_type == "random_forest") {
        rf_model_py <- joblib$load(paste0(filepath, ".pkl"))
        n_features <- rf_model_py$n_features_in_
      } else if (model_type %in% c("svmLinear", "svmRadial", "svmPoly")) {
        svm_model_py <- joblib$load(paste0(filepath, ".pkl"))
        n_features <- svm_model_py$n_features_in_
      } else {
        stop("Model type is not supported!")
      }
      if (is.null(n_features)) {
        stop("n_features_in_ could not be determined. Please define initial_type manually!")
      }
      FloatTensorType <- reticulate::import("skl2onnx.common.data_types")$FloatTensorType
      initial_type <- list(list("float_input", FloatTensorType(list(NULL, as.integer(n_features)))))
      if (model_type == "xgbTree") {
        onnx_model <- onnxmltools$convert_xgboost(xgb_model_py, initial_types = initial_type)
      } else if (model_type == "random_forest") {
        onnx_model <- onnxmltools$convert_sklearn(rf_model_py, initial_types = initial_type)
      } else if (model_type %in% c("svmLinear", "svmRadial", "svmPoly")) {
        onnx_model <- skl2onnx$convert_sklearn(svm_model_py, initial_types = initial_type)
      }
      onnx_file <- ensure_extension(filepath, "onnx")
      onnx$save_model(onnx_model, onnx_file)
      message(paste("Model successfully saved as ONNX under:", onnx_file))
      return(onnx_file)
    }

    save_model_as_mlm_stac_json <- function(model, filepath, tasks = list("classification"), options = list()) {
      mlm_stac_item <- list(
        type = "Feature",
        stac_version = "1.0.0",
        id = basename(sub("\\.json$", "", filepath)),
        properties = list(
          datetime = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ")
        ),
        geometry = NULL,
        bbox = NULL,
        stac_extensions = list(
          "https://stac-extensions.github.io/mlm/v1.0.0/schema.json"
        ),
        assets = list()
      )
      model_info <- list()
      if ("train" %in% class(model)) {
        model_info$"mlm:name" <- basename(sub("\\.json$", "", filepath))
        model_info$"mlm:architecture" <- model$modelInfo$label
        model_info$"mlm:tasks" <- tasks
        model_info$"mlm:framework" <- paste("R", paste(model$modelInfo$library, collapse = "/"), sep = " (")
        model_info$"mlm:framework_version" <- paste0("R ", R.version$major, ".", R.version$minor)
        model_type <- detect_model_type(model)
        if (model_type == "random_forest") {
          model_info$"mlm:total_parameters" <- model$finalModel$ntree * length(model$finalModel$forest$nodestatus)
          hyperparameters <- list()
          if (!is.null(model$finalModel$mtry)) hyperparameters$mtry <- model$finalModel$mtry
          if (!is.null(model$finalModel$ntree)) hyperparameters$ntree <- model$finalModel$ntree
          if (!is.null(model$finalModel$max_depth)) hyperparameters$max_depth <- model$finalModel$max_depth
        } else if (model_type %in% c("svmLinear", "svmRadial", "svmPoly")) {
          cost <- tryCatch(model$finalModel@kernelf@kpar$C, error = function(e) 1.0)
          nSV <- model$finalModel@nSV
          model_info$"mlm:total_parameters" <- nSV
          hyperparameters <- list(
            cost = cost,
            nSV = nSV,
            kernel = if (model_type == "svmLinear") "linear" else if (model_type == "svmRadial") "radial" else "poly"
          )
        } else if (model_type == "xgbTree") {
          nrounds <- model$finalModel$niter
          model_info$"mlm:total_parameters" <- nrounds
          hyperparameters <- list(
            max_depth = model$finalModel$tuneValue$max_depth,
            nrounds = nrounds,
            eta = model$finalModel$tuneValue$eta,
            gamma = model$finalModel$tuneValue$gamma
          )
        } else {
          hyperparameters <- list()
        }
        if (length(hyperparameters) > 0) model_info$"mlm:hyperparameters" <- hyperparameters
        model_info$"mlm:pretrained" <- FALSE
        model_info$"mlm:pretrained_source" <- NULL
        model_info$"mlm:batch_size_suggestion" <- 1
        model_info$"mlm:accelerator" <- NULL
        model_info$"mlm:accelerator_constrained" <- FALSE
        model_info$"mlm:accelerator_count" <- 1
        if (inherits(model$finalModel, "ksvm")) {
          predictors <- names(model$trainingData)[!names(model$trainingData) %in% ".outcome"]
        } else {
          predictors <- model$finalModel$xNames
        }
        model_info$"mlm:input" <- list(
          list(
            name = "Features",
            bands = predictors,
            input = list(
              shape = list(1, length(predictors)),
              dim_order = list("batch", "features"),
              data_type = "float32"
            ),
            description = "Input features for classification",
            pre_processing_function = NULL
          )
        )
        classes <- model$levels
        class_objects <- lapply(seq_along(classes), function(i) {
          list(
            value = i - 1,
            name = classes[i],
            description = paste("Class:", classes[i])
          )
        })
        model_info$"mlm:output" <- list(
          list(
            name = "CLASSIFICATION",
            tasks = tasks,
            result = list(
              shape = list(1, length(classes)),
              dim_order = list("batch", "classes"),
              data_type = "float32"
            ),
            description = "Predicted probabilities for classification",
            "classification:classes" = class_objects,
            post_processing_function = NULL
          )
        )
        if (length(options) > 0) {
          for (key in names(options)) {
            if (grepl("^mlm:", key)) {
              model_info[[key]] <- options[[key]]
            }
          }
        }
      } else {
        stop("Unknown model type: Please check the model!")
      }
      mlm_stac_item$properties <- c(mlm_stac_item$properties, model_info)
      rds_path <- ensure_extension(sub("\\.json$", "", filepath), "rds")
      saveRDS(model, file = rds_path)
      mlm_stac_item$assets <- list(
        model = list(
          href = rds_path,
          type = "application/octet-stream",
          title = paste(model_info$"mlm:architecture", "Model"),
          "mlm:artifact_type" = "R (RDS)",
          roles = list("mlm:model")
        )
      )
      jsonlite::write_json(mlm_stac_item, path = filepath, auto_unbox = TRUE, pretty = TRUE)
      message("Model was saved as MLM-STAC-JSON under: ", filepath)
      return(filepath)
    }


    save_model_as_mlm_stac_json_dl <- function(model, filepath, tasks = list("classification"), options = list()) {
      mlm_stac_item <- list(
        type = "Feature",
        stac_version = "1.0.0",
        id = basename(sub("\\.json$", "", filepath)),
        properties = list(
          datetime = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ")
        ),
        geometry = NULL,
        bbox = NULL,
        stac_extensions = list(
          "https://stac-extensions.github.io/mlm/v1.0.0/schema.json"
        ),
        assets = list()
      )
      model_info <- list()
      if ("nn_module" %in% class(model) || !is.null(model$conv_layers)) {
        model_info$"mlm:name" <- basename(sub("\\.json$", "", filepath))
        model_info$"mlm:architecture" <- "TempCNN"
        model_info$"mlm:tasks" <- tasks
        model_info$"mlm:framework" <- "R (torch)"
        model_info$"mlm:framework_version" <- paste0("R ", R.version$major, ".", R.version$minor)
        total_params <- sum(unlist(lapply(model$parameters, function(p) prod(dim(p)))))
        model_info$"mlm:total_parameters" <- as.integer(total_params)
        model_info$"mlm:hyperparameters" <- list(
          conv_layers = length(model$conv_layers),
          dense_layers = length(model$dense)
        )
        model_info$"mlm:pretrained" <- FALSE
        model_info$"mlm:pretrained_source" <- NULL
        model_info$"mlm:batch_size_suggestion" <- 1
        model_info$"mlm:accelerator" <- "gpu"
        model_info$"mlm:accelerator_constrained" <- FALSE
        model_info$"mlm:accelerator_count" <- 1
        input_channels <- if (!is.null(model$conv_layers[[1]])) model$conv_layers[[1]][[1]]$in_channels else NULL
        time_steps <- model$time_steps
        bands <- if (!is.null(model$input_data_columns)) model$input_data_columns else NULL
        if (is.null(input_channels)) stop("Could not extract input_channels from conv_layers!")
        if (is.null(time_steps)) stop("time_steps is missing in the model!")
        model_info$"mlm:input" <- list(
          list(
            name = "Temporal CNN Input",
            bands = if (!is.null(bands)) as.list(bands) else list("unknown"),
            input = list(
              shape = list(1, as.integer(input_channels), as.integer(time_steps)),
              dim_order = list("batch", "channels", "time_steps"),
              data_type = "float32"
            ),
            description = "Temporal input for CNN",
            pre_processing_function = NULL
          )
        )
        output_size <- if (!is.null(model$dense[[length(model$dense)]])) {
          model$dense[[length(model$dense)]]$out_features
        } else {
          NULL
        }
        if (is.null(output_size)) stop("Could not extract output_size from dense layer")
        model_info$"mlm:output" <- list(
          list(
            name = "CNN Output",
            tasks = tasks,
            result = list(
              shape = list(1, as.integer(output_size)),
              dim_order = list("batch", "features"),
              data_type = "float32"
            ),
            description = "Output features from TempCNN",
            "classification:classes" = if (tasks[[1]] == "classification") {
              lapply(0:(output_size - 1), function(i) list(
                value = i,
                name = paste("class", i),
                description = paste("Class", i)
              ))
            } else NULL,
            post_processing_function = NULL
          )
        )
        if (length(options) > 0) {
          for (key in names(options)) {
            if (grepl("^mlm:", key)) {
              model_info[[key]] <- options[[key]]
            }
          }
        }
      } else {
        stop("Unknown model type: Please check the model!")
      }
      mlm_stac_item$properties <- c(mlm_stac_item$properties, model_info)

      rds_path <- ensure_extension(sub("\\.json$", "", filepath), "rds")
      con <- rawConnection(raw(0), "wb")
      torch::torch_save(model, con)
      raw_model <- rawConnectionValue(con)
      close(con)
      saveRDS(raw_model, file = rds_path)

      mlm_stac_item$assets <- list(
        model = list(
          href = rds_path,
          type = "application/octet-stream",
          title = "TempCNN Model",
          "mlm:artifact_type" = "R (Raw RDS)",
          roles = list("mlm:model")
        )
      )
      jsonlite::write_json(mlm_stac_item, path = filepath, auto_unbox = TRUE, pretty = TRUE)
      message("Model was saved as MLM-STAC-JSON under: ", filepath)
      return(filepath)


    }

    name_base <- name
    message(name_base)
    if (!is.null(tasks) && length(tasks) > 0) {
      options <- list()
      options[["mlm:tasks"]] <- tasks
    }
    result <- list()
    tmp <- shared_dir

    if(is.character(data) && endsWith(data, ".pt")){
      library(torch)
      data <- torch::torch_load(data)
    }

    if ("nn_module" %in% class(data) || !is.null(data$conv_layers)) {
      message("Detected a Torch model. Using deep learning conversion...")
      torch_path <- file.path(tmp, ensure_extension(name_base, "pt"))
      model_info <- save_torch_model(data, torch_path)
      onnx_path <- convert_r_torch_to_onnx(data, file.path(tmp, name_base))
      result$onnx <- onnx_path
      result$torch <- torch_path
      json_file <- file.path(tmp, ensure_extension(name_base, "json"))
      save_model_as_mlm_stac_json_dl(data, json_file, tasks, options)

      result$json <- json_file
    } else if ("train" %in% class(data)) {
      message("Machine model detected...")
      model_type <- detect_model_type(data)
      message("Detected model type: ", model_type)
      convert_model_to_pkl(data, model_type, file.path(tmp, name_base))
      onnx_path <- save_ml_model_as_onnx(model_type, file.path(tmp, name_base))
      result$onnx <- onnx_path
      json_file <- file.path(tmp, ensure_extension(name_base, "json"))
      save_model_as_mlm_stac_json(data, json_file, tasks, options)
      result$json <- json_file
    } else {
      stop("Unknown model type: must be 'nn_module' (Torch) or 'train' (Caret).")
    }

    rds_path <- file.path(tmp, ensure_extension(name_base, "rds"))
    result$rds <- rds_path

    if (length(options) > 0) {
      add_metadata_to_onnx(result$onnx, options)
    }

    download_base <- Sys.getenv("DOWNLOAD_BASE_URL", "http://localhost:8000/download/")
    download_links <- list(
      onnx  = sprintf("%s%s", download_base, basename(result$onnx)),
      json  = sprintf("%s%s", download_base, basename(result$json)),
      rds   = sprintf("%s%s", download_base, basename(result$rds))
    )
    if (!is.null(result$torch)) {
      download_links$torch <- sprintf("%s%s", download_base, basename(result$torch))
    }
    message("Model exported successfully (Download-Links):")
    message("- ONNX: ", download_links$onnx)
    message("- JSON (MLM-STAC): ", download_links$json)
    message("- RDS: ", download_links$rds)
    if (!is.null(download_links$torch)) {
      message("- TorchScript: ", download_links$torch)
    }
    return(result)
  }
)

#######################################################
#' Download a file from the shared temporary directory
#'
#' @description
#' Serves a file stored in the shared temporary directory as an HTTP download.
#' Checks that the filename is provided and exists, sets appropriate headers
#' (including Content-Type and Content-Disposition), and returns the file's binary
#' content or an error status.
#'
#' @param filename Character. Name of the file to download (relative to `SHARED_TEMP_DIR`).
#' @param res Response. Plumber response object that will be modified with status,
#'            headers, and body.
#'
#' @return
#' The modified `res` object containing either:
#' - On success: HTTP 200 status, download headers, and raw binary body of the file.
#' - On error: HTTP 400 or 404 status and a list with an `error` message.
#'
download <- function(filename, res) {
  if (is.null(filename) || filename == "") {
    res$status <- 400
    return(list(error = "No file name specified"))
  }

  shared_dir <- Sys.getenv("SHARED_TEMP_DIR", tempdir())
  file_path <- file.path(shared_dir, filename)

  if (!file.exists(file_path)) {
    res$status <- 404
    return(list(error = "File not found"))
  }

  ext <- tools::file_ext(filename)
  content_type <- switch(
    ext,
    "json" = "application/json",
    "rds" = "application/octet-stream",
    "onnx" = "application/octet-stream",
    "pt"   = "application/octet-stream",
    "pkl"  = "application/octet-stream",
    "bin"  = "application/octet-stream",
    "txt"  = "text/plain",
    "application/octet-stream"
  )

  res$setHeader("Content-Type", content_type)
  res$setHeader("Content-Disposition", sprintf('attachment; filename="%s"', filename))

  file_content <- readBin(file_path, "raw", n = file.info(file_path)$size)
  res$body <- file_content

  return(res)
}
