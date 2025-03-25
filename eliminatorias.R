# ================================
# Librerías necesarias
# ================================
library(tidyverse)
library(lubridate)
library(caret)
library(randomForest)

# ================================
# Cargar y preparar datos
# ================================
datos <- read.csv("datos.csv")

# Asegurémonos que la fecha esté en formato Date
datos$fecha <- as.Date(datos$fecha)

# Colores de los equipos
colores_equipos <- c(
  "Argentina" = "#A8DADD",
  "Bolivia" = "#4CAF50",
  "Brasil" = "#9BC53D",
  "Chile" = "#C72C68",
  "Colombia" = "#EDE8D2",
  "Ecuador" = "#FEC20E",
  "Peru" = "#D50028",
  "Paraguay" = "#D32F0F",
  "Uruguay" = "#6EC6F9",
  "Venezuela" = "#8B0031"
)

# ================================
# Análisis Exploratorio de Datos (EDA)
# ================================
# Estadísticas descriptivas básicas
summary_estadisticas <- summary(datos[, c("goles_a_favor", "goles_en_contra", "puntos", 
                                          "disparos", "disparos_al_arco", "posesion", 
                                          "pases", "pases_completados_en_porcentaje", 
                                          "tarjetas_amarillas", "tarjetas_rojas")])
print(summary_estadisticas)

# Distribución de resultados
tabla_resultados <- table(datos$resultado_texto)
print(tabla_resultados)

# Promedio de posesión por equipo
posesion_por_equipo <- datos %>%
  group_by(equipo) %>%
  summarise(posesion = mean(posesion, na.rm = TRUE)) %>%
  arrange(desc(posesion))
print(posesion_por_equipo)

# Promedio de disparos por equipo
disparos_por_equipo <- datos %>%
  group_by(equipo) %>%
  summarise(disparos_promedio = mean(disparos, na.rm = TRUE)) %>%
  arrange(desc(disparos_promedio))
print(disparos_por_equipo)

# Gráfico de goles a favor vs en contra por equipo
datos_resumidos <- datos %>%
  group_by(equipo) %>%
  summarise(goles_a_favor = sum(goles_a_favor),
            goles_en_contra = sum(goles_en_contra))

ggplot(datos_resumidos, aes(x = reorder(equipo, goles_a_favor))) +
  geom_bar(aes(y = goles_a_favor), stat = "identity", fill = "blue", alpha = 0.7) +
  geom_bar(aes(y = goles_en_contra), stat = "identity", fill = "red", alpha = 0.7) +
  labs(title = "Goles a favor vs Goles en contra",
       y = "Cantidad de goles",
       x = "Equipo") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ================================
# Insights
# ================================
# Efectividad de alineaciones
alineacion_efectividad <- datos %>%
  group_by(equipo, alineacion) %>%
  summarise(partidos = n(),
            victorias = sum(resultado_texto == "Victoria"),
            puntos_total = sum(puntos),
            efectividad_porcentaje = (puntos_total / (partidos * 3)) * 100,
            .groups = "drop") %>%
  arrange(equipo, desc(efectividad_porcentaje))

# Mejor alineación por equipo
mejor_alineacion_por_equipo <- alineacion_efectividad %>%
  group_by(equipo) %>%
  filter(partidos >= 2) %>%
  slice_max(order_by = efectividad_porcentaje, n = 1)
print(mejor_alineacion_por_equipo)

# Cambios de entrenador y formación por equipo
cambios_entrenador <- datos %>%
  group_by(equipo) %>%
  summarise(entrenadores_diferentes = n_distinct(entrenador), .groups = "drop") %>%
  arrange(desc(entrenadores_diferentes))

cambios_formacion <- datos %>%
  group_by(equipo) %>%
  summarise(formaciones_diferentes = n_distinct(alineacion), .groups = "drop") %>%
  arrange(desc(formaciones_diferentes))

# Comparación de cambios
equipo_menos_cambios_entrenador <- cambios_entrenador %>% 
  slice_min(order_by = entrenadores_diferentes, n = 1)

print("Equipo con menos cambios de entrenador:")
print(equipo_menos_cambios_entrenador)

# Equipo con mayor porcentaje de pases completados
pases_completados_por_equipo <- datos %>%
  group_by(equipo) %>%
  summarise(pases_completados_prom = mean(pases_completados_en_porcentaje, na.rm = TRUE)) %>%
  arrange(desc(pases_completados_prom))

print(paste("Equipo con mayor % de pases completados:", 
            pases_completados_por_equipo$equipo[1], 
            "con", round(pases_completados_por_equipo$pases_completados_prom, 1), "% en promedio"))

# ================================
# Modelado Predictivo
# ================================
# Preparar datos para modelado
datos_modelado <- datos %>%
  select(equipo, rival, 
         goles_a_favor, 
         goles_en_contra, 
         posesion, 
         disparos, 
         disparos_al_arco, 
         pases_completados_en_porcentaje, 
         altitud)

# Función para crear características para cada partido
preparar_datos_partido <- function(equipo_local, equipo_visitante, datos) {
  stats_local <- datos %>%
    filter(equipo == equipo_local) %>%
    summarise(across(c(goles_a_favor, goles_en_contra, posesion, disparos, disparos_al_arco, 
                       pases_completados_en_porcentaje), mean, na.rm = TRUE))
  
  stats_visitante <- datos %>%
    filter(equipo == equipo_visitante) %>%
    summarise(across(c(goles_a_favor, goles_en_contra, posesion, disparos, disparos_al_arco, 
                       pases_completados_en_porcentaje), mean, na.rm = TRUE))
  
  altitud_local <- datos %>%
    filter(equipo == equipo_local) %>%
    select(altitud) %>%
    slice(1) %>%
    pull(altitud)
  
  partido <- data.frame(
    Equipo_Local = equipo_local,
    Equipo_Visitante = equipo_visitante,
    Goles_a_favor_local_prom = stats_local$goles_a_favor,
    Goles_en_contra_local_prom = stats_local$goles_en_contra,
    Posesion_local_prom = stats_local$posesion,
    Disparos_local_prom = stats_local$disparos,
    Disparos_al_Arco_local_prom = stats_local$disparos_al_arco,
    Pases_completados_local_prom = stats_local$pases_completados_en_porcentaje,
    Goles_a_favor_visitante_prom = stats_visitante$goles_a_favor,
    Goles_en_contra_visitante_prom = stats_visitante$goles_en_contra,
    Posesion_visitante_prom = stats_visitante$posesion,
    Disparos_visitante_prom = stats_visitante$disparos,
    Disparos_al_Arco_visitante_prom = stats_visitante$disparos_al_arco,
    Pases_completados_visitante_prom = stats_visitante$pases_completados_en_porcentaje,
    Altitud = altitud_local
  )
  
  return(partido)
}

# Crear datos históricos para el modelo
partidos_historicos <- data.frame()
for (i in 1:nrow(datos)) {
  partido <- preparar_datos_partido(datos$equipo[i], datos$rival[i], datos)
  partido$goles_local <- datos$goles_a_favor[i]
  partido$goles_visitante <- datos$goles_en_contra[i]
  partidos_historicos <- rbind(partidos_historicos, partido)
}

# Dividir los datos en entrenamiento y validación
set.seed(123)
indices_train <- createDataPartition(partidos_historicos$goles_local, p = 0.8, list = FALSE)
train_data <- partidos_historicos[indices_train, ]
test_data <- partidos_historicos[-indices_train, ]

# Entrenar modelos de regresión
modelo_goles_local <- randomForest(
  goles_local ~ Goles_a_favor_local_prom + Goles_en_contra_local_prom + 
    Posesion_local_prom + Disparos_local_prom + Disparos_al_Arco_local_prom +
    Goles_a_favor_visitante_prom + Goles_en_contra_visitante_prom +
    Posesion_visitante_prom + Disparos_visitante_prom + Altitud,
  data = train_data, ntree = 100
)

modelo_goles_visitante <- randomForest(
  goles_visitante ~ Goles_a_favor_visitante_prom + Goles_en_contra_visitante_prom + 
    Posesion_visitante_prom + Disparos_visitante_prom + Disparos_al_Arco_visitante_prom +
    Goles_a_favor_local_prom + Goles_en_contra_local_prom +
    Posesion_local_prom + Disparos_local_prom + Altitud,
  data = train_data, ntree = 100
)

# Evaluación de los modelos
pred_local <- predict(modelo_goles_local, test_data)
pred_visitante <- predict(modelo_goles_visitante, test_data)

# Error cuadrático medio
rmse_local <- sqrt(mean((pred_local - test_data$goles_local)^2))
rmse_visitante <- sqrt(mean((pred_visitante - test_data$goles_visitante)^2))

cat("RMSE para predicción de goles locales:", rmse_local, "\n")
cat("RMSE para predicción de goles visitantes:", rmse_visitante, "\n")

# Predicciones para la siguiente jornada
proximos_partidos <- list(
  c("Bolivia", "Uruguay"),
  c("Chile", "Ecuador"),
  c("Venezuela", "Peru"),
  c("Colombia", "Paraguay"),
  c("Argentina", "Brazil")
)

predicciones <- data.frame()
for (partido in proximos_partidos) {
  local <- partido[1]
  visitante <- partido[2]
  
  datos_partido <- preparar_datos_partido(local, visitante, datos)
  
  goles_local_pred <- predict(modelo_goles_local, datos_partido)
  goles_visitante_pred <- predict(modelo_goles_visitante, datos_partido)
  
  # Redondeamos las predicciones
  goles_local_pred <- round(goles_local_pred)
  goles_visitante_pred <- round(goles_visitante_pred)
  
  # Determinamos el resultado
  if (goles_local_pred > goles_visitante_pred) {
    resultado <- "Victoria Local"
    puntos_local <- 3
    puntos_visitante <- 0
  } else if (goles_local_pred < goles_visitante_pred) {
    resultado <- "Victoria Visitante"
    puntos_local <- 0
    puntos_visitante <- 3
  } else {
    resultado <- "Empate"
    puntos_local <- 1
    puntos_visitante <- 1
  }
  
  prediccion <- data.frame(
    Local = local,
    Visitante = visitante,
    Goles_Local_Pred = goles_local_pred,
    Goles_Visitante_Pred = goles_visitante_pred,
    Resultado_Pred = paste0(goles_local_pred, "-", goles_visitante_pred, " (", resultado, ")"),
    Puntos_Local = puntos_local,
    Puntos_Visitante = puntos_visitante
  )
  
  predicciones <- rbind(predicciones, prediccion)
}

print("Predicciones para la siguiente jornada:")
print(predicciones)


# ================================
# Puntos Acumulados por Equipo a lo Largo de las Jornadas
# ================================


# Gráfico de Puntos Acumulados

grafico_puntos_acumulados <- ggplot(datos, aes(x = factor(jornada, 1:13), y = puntos_acumulados, color = equipo, group = equipo)) + 
  geom_line(size = 1.2, alpha = 0.8) + 
  geom_point(size = 3) + 
  geom_text(data = subset(datos, jornada == 13), 
            aes(label = round(puntos_acumulados, 0)), 
            nudge_x = 0.3, 
            show.legend = FALSE) +  # Mostrar números solo en la jornada 13
  scale_color_manual(values = colores_equipos) + 
  scale_y_continuous(breaks = seq(0, 30, by = 3)) + 
  labs(
    title = "Puntos Acumulados por Equipo", 
    x = "Jornada", 
    y = "Puntos Acumulados"
  ) + 
  theme_minimal() + 
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1), 
    legend.position = "right", 
    plot.title = element_text(hjust = 0.5, face = "bold"), 
    panel.grid.minor = element_blank()
  ) + 
  guides(color = guide_legend(title = "Equipos"))

# Imprimir gráfico
print(grafico_puntos_acumulados)

# Guardar gráfico
ggsave("puntos_acumulados_linechart.png", grafico_puntos_acumulados, width = 12, height = 7, dpi = 300)