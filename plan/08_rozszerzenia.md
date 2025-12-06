# 8) Rozszerzenia (opcjonalne)
- Segmentacja (Mask R-CNN lub YOLO-seg) dla dokładniejszych pomiarów szerokości i detekcji uszkodzeń konturowych.
- Śledzenie (SORT/DeepSORT/ByteTrack) dla stabilniejszych cykli i przypisywania detekcji do elementów.
- Alerty w czasie rzeczywistym: webhook/email/SMS; buforowanie detekcji w Redis dla web GUI.
- Optymalizacja wydajności: ONNX/TensorRT dla inference na GPU, multithreading wideo + batchowanie klatek.
- Monitoring: metryki Prometheus + prosty dashboard (Grafana) do uptime i statystyk alertów.
- Hardening produkcyjny: walidacja wejść API, limity wielkości wideo, obsługa błędów i time-outów.
