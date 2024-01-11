import csv
import os

def write_model_peformance(output_path, model_name, batch_idx, confidence_threshold, vehicle_accuracies, peds_accuracies):
    # n_of_dirs = len(os.listdir(output_path)) + 1
    with open(f'{output_path}/{model_name}_statistics.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(['model_name', "batch_idx", "Confidence threshold %" "Vehicle average accuracy %", "Pedestrians average accuracy %"])
        for (conf, vehicle_accuracy), (conf, peds_accuracy) in zip(vehicle_accuracies.items(), peds_accuracies.items()):
            vehicle_avg_accuracy = round(sum(vehicle_accuracy)/len(vehicle_accuracy), 2)
            peds_avg_accuracy = round(sum(peds_accuracy)/len(peds_accuracy), 2)
            writer.writerow([model_name, batch_idx, conf, vehicle_avg_accuracy, peds_avg_accuracy])
        writer.writerow([])

def write_ensemble_model_peformance(output_path, model_name, batch_idx, vehicle_accuracies, peds_accuracies):
    # n_of_dirs = len(os.listdir(output_path)) + 1
    with open(f'{output_path}/{model_name}_statistics.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(['model_name', "batch_idx", "Vehicle average accuracy %", "Pedestrians average accuracy %"])
        vehicle_avg_accuracy = round(sum(vehicle_accuracies)/len(vehicle_accuracies), 2)
        peds_avg_accuracy = round(sum(peds_accuracies)/len(peds_accuracies), 2)
        writer.writerow([model_name, batch_idx, vehicle_avg_accuracy, peds_avg_accuracy])
        writer.writerow([])


