#!/usr/bin/env python3
"""
Simulation WNTR avec export InfluxDB pour visualisation Grafana
Combine la logique de simulation WNTR avec l'export de données graphiques vers InfluxDB
Version corrigée avec enregistrement complet des itérations
"""

import wntr
import wntr.network.controls as controls
import csv
import sys
import pandas as pd
import yaml
import time
import os
from influxdb import InfluxDBClient
from datetime import datetime
import argparse
import re

class WNTRInfluxSimulation:
    def __init__(self, config_file_path):
        # Chargement de la configuration
        self.config_options = self.load_config(config_file_path)

        # Configuration de simulation
        self.week_index = int(self.config_options.get('week_index', 0))
        self.output_path = self.config_options['output_ground_truth_path']
        self.simulation_days = int(self.config_options['duration_days'])

        # Configuration InfluxDB
        self.influx_config = self.config_options.get('influxdb', {})
        self.influx_host = self.influx_config.get('host', 'localhost')
        self.influx_port = self.influx_config.get('port', 8086)
        self.influx_username = self.influx_config.get('username', '')
        self.influx_password = self.influx_config.get('password', '')
        self.influx_database = self.influx_config.get('database', 'water2')

        # Initialisation du réseau WNTR
        inp_file = self.config_options['inp_file']
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.inp_file = inp_file

        # Listes des composants
        self.node_list = list(self.wn.node_name_list)
        self.link_list = list(self.wn.link_name_list)

        self.tank_list = self.get_node_list_by_type(self.node_list, 'Tank')
        self.junction_list = self.get_node_list_by_type(self.node_list, 'Junction')
        self.reservoir_list = self.get_node_list_by_type(self.node_list, 'Reservoir')
        self.pump_list = self.get_link_list_by_type(self.link_list, 'Pump')
        self.valve_list = self.get_link_list_by_type(self.link_list, 'Valve')
        self.pipe_list = self.get_link_list_by_type(self.link_list, 'Pipe')

        # Coordonnées des nœuds
        self.coordinates = self.extract_coordinates()

        # Mapping des couleurs pour Grafana
        self.color_mapping = {
            'junction': '#3498db',    # Bleu
            'reservoir': '#e74c3c',   # Rouge
            'tank': '#f39c12',        # Orange
            'pipe': '#2ecc71',        # Vert
            'pump': '#9b59b6',        # Violet
            'valve': '#e67e22'        # Orange foncé
        }

        # Mapping des tailles
        self.size_mapping = {
            'junction': 8.0,
            'reservoir': 25.0,
            'tank': 18.0
        }

        # Initialisation des résultats CSV
        self.results_list = []
        self.init_csv_headers()

        # Initialisation de la simulation
        self.initialize_simulation()

        # Configuration du simulateur
        simulator_string = self.config_options.get('simulator', 'dd')
        if simulator_string == 'pdd':
            print('Running simulation using PDD')
            self.wn.options.hydraulic.demand_model = 'PDD'
        elif simulator_string == 'dd':
            print('Running simulation using DD')
        else:
            print('Invalid simulation mode, using DD by default')

        self.sim = wntr.sim.WNTRSimulator(self.wn)

        # Initialisation du client InfluxDB
        self.influx_client = InfluxDBClient(
            host=self.influx_host,
            port=self.influx_port,
            username=self.influx_username,
            password=self.influx_password,
            database=self.influx_database
        )

        print(f"Starting simulation for {inp_file}")
        print(f"InfluxDB: {self.influx_host}:{self.influx_port}/{self.influx_database}")

    def load_config(self, config_path):
        """Charge le fichier de configuration YAML"""
        with open(config_path) as config_file:
            options = yaml.load(config_file, Loader=yaml.FullLoader)
        return options

    def extract_coordinates(self):
        """Extrait les coordonnées des nœuds depuis le réseau WNTR"""
        coordinates = {}
        for node_name in self.node_list:
            node = self.wn.get_node(node_name)
            if hasattr(node, 'coordinates') and node.coordinates:
                coordinates[node_name] = {
                    'x': float(node.coordinates[0]),
                    'y': float(node.coordinates[1])
                }
            else:
                # Coordonnées par défaut si non disponibles
                coordinates[node_name] = {'x': 0.0, 'y': 0.0}
        return coordinates

    def get_node_list_by_type(self, a_list, a_type):
        """Filtre les nœuds par type"""
        return [str(node) for node in a_list if self.wn.get_node(node).node_type == a_type]

    def get_link_list_by_type(self, a_list, a_type):
        """Filtre les liens par type"""
        return [str(link) for link in a_list if self.wn.get_link(link).link_type == a_type]

    def init_csv_headers(self):
        """Initialise les en-têtes CSV"""
        headers = ["Timestamps"]
        headers.extend([tank + "_LEVEL" for tank in self.tank_list])
        headers.extend([junction + "_PRESSURE" for junction in self.junction_list])

        for pump in self.pump_list:
            headers.extend([pump + "_FLOW", pump + "_STATUS"])
        for valve in self.valve_list:
            headers.extend([valve + "_FLOW", valve + "_STATUS"])

        self.results_list.append(headers)

    def initialize_simulation(self):
        """Initialise les conditions de simulation"""
        if 'initial_custom_flag' in self.config_options:
            custom_initial_conditions_flag = bool(self.config_options['initial_custom_flag'])
            if custom_initial_conditions_flag:
                self.setup_custom_initial_conditions()

    def setup_custom_initial_conditions(self):
        """Configure les conditions initiales personnalisées"""
        try:
            demand_patterns_path = self.config_options['demand_patterns_path']
            starting_demand_path = self.config_options['starting_demand_path']
            initial_tank_levels_path = self.config_options['initial_tank_levels_path']

            print(f"Running simulation with week index: {self.week_index}")

            total_demands = pd.read_csv(demand_patterns_path, index_col=0)
            demand_starting_points = pd.read_csv(starting_demand_path, index_col=0)
            initial_tank_levels = pd.read_csv(initial_tank_levels_path, index_col=0)

            # Définir la limite selon la durée
            limit = 167 if self.simulation_days == 7 else 239

            week_start = demand_starting_points.iloc[self.week_index][0]
            week_demands = total_demands.loc[week_start:week_start + limit, :]

            # Appliquer les patterns de demande
            for name, pat in self.wn.patterns():
                if name in week_demands.columns:
                    pat.multipliers = week_demands[name].values.tolist()

            # Appliquer les niveaux initiaux des réservoirs
            for tank in self.tank_list:
                if tank in initial_tank_levels.columns:
                    self.wn.get_node(tank).init_level = float(initial_tank_levels.iloc[self.week_index][tank])

        except Exception as e:
            print(f"Warning: Could not load custom initial conditions: {e}")

    def _safe_float(self, value, default=0.0):
        """Convertit une valeur en float de manière sécurisée"""
        try:
            if value == '*' or value == '' or value is None:
                return float(default)
            return float(value)
        except (ValueError, TypeError):
            return float(default)

    def _calculate_node_stats(self, node_name, node_type):
        """Calcule les statistiques d'affichage pour un nœud"""
        node = self.wn.get_node(node_name)

        if node_type == 'junction':
            # Gestion sécurisée des valeurs None
            head = getattr(node, 'head', 0.0) or 0.0
            elevation = getattr(node, 'elevation', 0.0) or 0.0
            main_stat = head - elevation  # Pression
            second_stat = elevation
            node_radius = self.size_mapping['junction']
        elif node_type == 'reservoir':
            main_stat = getattr(node, 'head', 0.0) or 0.0
            second_stat = 0.0
            node_radius = self.size_mapping['reservoir']
        elif node_type == 'tank':
            main_stat = getattr(node, 'level', 0.0) or 0.0
            second_stat = getattr(node, 'elevation', 0.0) or 0.0
            node_radius = max(10.0, min(30.0, getattr(node, 'diameter', 10.0) / 3.0))
        else:
            main_stat = 0.0
            second_stat = 0.0
            node_radius = 10.0

        return main_stat, second_stat, node_radius

    def export_to_influxdb(self, iteration, timestamp_offset=0):
        """Exporte l'état actuel vers InfluxDB"""
        # Créer un timestamp basé sur l'itération pour assurer l'ordre
        base_time = datetime.utcnow()
        timestamp = datetime.utcfromtimestamp(
            base_time.timestamp() + iteration + timestamp_offset
        ).strftime('%Y-%m-%dT%H:%M:%SZ')

        nodes_data = []
        edges_data = []

        # Export des nœuds
        all_nodes = {**{n: 'junction' for n in self.junction_list},
                     **{n: 'reservoir' for n in self.reservoir_list},
                     **{n: 'tank' for n in self.tank_list}}

        for node_id, node_type in all_nodes.items():
            coords = self.coordinates.get(node_id, {'x': 0.0, 'y': 0.0})
            main_stat, second_stat, node_radius = self._calculate_node_stats(node_id, node_type)

            # Déterminer si le nœud doit être mis en évidence
            highlighted = 0.0
            if node_type == 'tank':
                tank = self.wn.get_node(node_id)
                min_level = getattr(tank, 'min_level', None)
                max_level = getattr(tank, 'max_level', None)
                current_level = getattr(tank, 'level', 0.0) or 0.0

                if min_level is not None and max_level is not None:
                    if current_level <= min_level or current_level >= max_level:
                        highlighted = 1.0

            fields = {
                "x_coord": float(coords['x']),
                "y_coord": float(coords['y']),
                "title": str(node_type.capitalize()),
                "subtitle": f"ID: {node_id}",
                "mainStat": float(main_stat),
                "secondaryStat": float(second_stat),
                "noderadius": float(node_radius),
                "highlighted": float(highlighted),
                "color": str(self.color_mapping.get(node_type, '#95a5a6')),
                "iteration": float(iteration)
            }

            point = {
                "measurement": "nodes",
                "tags": {
                    "id": str(node_id),
                    "type": str(node_type),
                },
                "time": timestamp,
                "fields": fields
            }
            nodes_data.append(point)

        # Export des liens
        all_links = {**{l: 'pipe' for l in self.pipe_list},
                     **{l: 'pump' for l in self.pump_list},
                     **{l: 'valve' for l in self.valve_list}}

        for link_id, link_type in all_links.items():
            link = self.wn.get_link(link_id)

            # Gestion sécurisée du flow
            flow = getattr(link, 'flow', 0.0) or 0.0

            # Statistiques selon le type
            if link_type == 'pipe':
                main_stat = getattr(link, 'diameter', 0.0) or 0.0
                second_stat = getattr(link, 'length', 0.0) or 0.0
                thickness = max(1.0, min(10.0, main_stat / 10.0)) if main_stat > 0 else 2.0
                highlighted = 0.0
            elif link_type == 'pump':
                main_stat = abs(flow)
                second_stat = 0.0
                thickness = 4.0
                status = link.status if isinstance(link.status, int) else getattr(link.status, 'value', 0)
                highlighted = 1.0 if status == 1 else 0.0
            elif link_type == 'valve':
                main_stat = getattr(link, 'diameter', 0.0) or 0.0
                second_stat = abs(flow)
                thickness = max(1.0, min(8.0, main_stat / 15.0)) if main_stat > 0 else 2.0
                status = link.status if isinstance(link.status, int) else getattr(link.status, 'value', 0)
                highlighted = 1.0 if status != 1 else 0.0
            else:
                main_stat = abs(flow)
                second_stat = 0.0
                thickness = 2.0
                highlighted = 0.0

            # Couleur selon l'état
            color = self.color_mapping.get(link_type, '#95a5a6')
            if link_type == 'pump':
                status = link.status if isinstance(link.status, int) else getattr(link.status, 'value', 0)
                color = '#2ecc71' if status == 1 else '#e74c3c'
            elif link_type == 'valve':
                status = link.status if isinstance(link.status, int) else getattr(link.status, 'value', 0)
                color = '#9b59b6' if status == 1 else '#e67e22'

            fields = {
                "mainStat": float(main_stat),
                "secondaryStat": float(second_stat),
                "thickness": float(thickness),
                "highlighted": float(highlighted),
                "color": str(color),
                "flow": float(flow),
                "iteration": float(iteration)
            }

            # Ajouter le statut pour pompes et vannes
            if link_type in ['pump', 'valve']:
                status = link.status if isinstance(link.status, int) else getattr(link.status, 'value', 0)
                fields["status"] = float(status)

            point = {
                "measurement": "edges",
                "tags": {
                    "id": str(link_id),
                    "type": str(link_type),
                    "source": str(link.start_node_name),
                    "target": str(link.end_node_name),
                },
                "time": timestamp,
                "fields": fields
            }
            edges_data.append(point)

        # Envoi vers InfluxDB
        try:
            if nodes_data:
                self.influx_client.write_points(nodes_data)
            if edges_data:
                self.influx_client.write_points(edges_data)
            print(f"Iteration {iteration}: Data exported to InfluxDB ({len(nodes_data)} nodes, {len(edges_data)} edges)")
        except Exception as e:
            print(f"Error writing to InfluxDB at iteration {iteration}: {e}")

    def register_results(self, results, iteration):
        """Enregistre les résultats pour le CSV"""
        # Utiliser le timestamp depuis les résultats ou créer un timestamp basé sur l'itération
        if hasattr(results, 'timestamp') and results.timestamp:
            timestamp = results.timestamp
        else:
            # Créer un timestamp basé sur l'itération
            timestamp = f"iteration_{iteration:06d}"

        values_list = [timestamp]

        # Niveaux des réservoirs
        for tank in self.tank_list:
            tank_obj = self.wn.get_node(tank)
            level = getattr(tank_obj, 'level', 0.0) or 0.0
            values_list.append(level)

        # Pressions des jonctions
        for junction in self.junction_list:
            junction_obj = self.wn.get_node(junction)
            head = getattr(junction_obj, 'head', 0.0) or 0.0
            elevation = getattr(junction_obj, 'elevation', 0.0) or 0.0
            pressure = head - elevation
            values_list.append(pressure)

        # États des pompes
        for pump in self.pump_list:
            pump_obj = self.wn.get_link(pump)
            flow = getattr(pump_obj, 'flow', 0.0) or 0.0
            values_list.append(flow)
            status = pump_obj.status if isinstance(pump_obj.status, int) else pump_obj.status.value
            values_list.append(status)

        # États des vannes
        for valve in self.valve_list:
            valve_obj = self.wn.get_link(valve)
            flow = getattr(valve_obj, 'flow', 0.0) or 0.0
            values_list.append(flow)
            status = valve_obj.status if isinstance(valve_obj.status, int) else valve_obj.status.value
            values_list.append(status)

        self.results_list.append(values_list)
        return values_list

    def register_initial_state(self):
        """Enregistre l'état initial (itération 0) avant la simulation"""
        print("Recording initial state (iteration 0)...")

        # Créer un objet résultat fictif pour l'état initial
        class InitialResults:
            def __init__(self):
                self.timestamp = "iteration_000000"

        initial_results = InitialResults()
        self.register_results(initial_results, 0)
        self.export_to_influxdb(0, timestamp_offset=-1)  # Timestamp antérieur pour l'état initial

    def write_results(self):
        """Sauvegarde les résultats CSV"""
        try:
            output_dir = 'output'
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, self.output_path)

            print(f"Writing results to {output_file}...")
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.results_list)
            print(f"Results saved successfully! {len(self.results_list)-1} data rows written.")
        except Exception as e:
            print(f"Error writing CSV file: {e}")

    def run_simulation(self):
        """Exécute la simulation principale"""
        # Calcul du nombre d'itérations
        iteration_limit = int((self.simulation_days * 24 * 3600) / self.wn.options.time.hydraulic_timestep)

        print(f"Simulation will run for {self.simulation_days} days.")
        print(f"Hydraulic timestep is {self.wn.options.time.hydraulic_timestep} seconds")
        print(f"Total iterations expected: {iteration_limit}")
        print("=" * 60)

        # CORRECTION 1: Enregistrer l'état initial AVANT de commencer la simulation
        self.register_initial_state()

        iteration = 0
        simulation_completed = False

        try:
            # CORRECTION 2: Boucle continue de 0 à iteration_limit-1
            while iteration < iteration_limit:
                try:
                    print(f"ITERATION {iteration + 1}/{iteration_limit} ------------- ", flush=True)

                    # Configurer la durée pour un pas de temps
                    self.wn.options.time.duration = self.wn.options.time.hydraulic_timestep

                    # Exécuter la simulation
                    results = self.sim.run_sim(convergence_error=True)

                    # CORRECTION 3: Numérotation cohérente des itérations
                    iteration_number = iteration + 1

                    # Enregistrer les résultats avec le bon numéro d'itération
                    self.register_results(results, iteration_number)

                    # Exporter vers InfluxDB avec le bon numéro d'itération
                    self.export_to_influxdb(iteration_number)

                    iteration += 1

                    # Petite pause pour éviter la surcharge
                    time.sleep(5)

                except KeyboardInterrupt:
                    print("\n\nUser interruption (Ctrl+C). Stopping simulation...")
                    break
                except Exception as e:
                    print(f"Error at iteration {iteration + 1}: {e}")
                    # Afficher plus de détails sur l'erreur pour le debug
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")

                    # CORRECTION 4: Continuer avec l'itération suivante au lieu de s'arrêter
                    print(f"Skipping iteration {iteration + 1} due to error, continuing...")
                    iteration += 1
                    continue

            simulation_completed = (iteration >= iteration_limit)

            if simulation_completed:
                print("\n" + "=" * 60)
                print("SIMULATION COMPLETED SUCCESSFULLY!")
            else:
                print("\n" + "=" * 60)
                print("SIMULATION INTERRUPTED")

            print(f"Total iterations executed: {iteration + 1}")  # +1 pour inclure l'état initial
            print(f"Data rows collected: {len(self.results_list)-1}")

        except Exception as e:
            print(f"Critical error in simulation: {e}")

        finally:
            print("\nSaving results...")
            self.write_results()

            print("Cleaning up resources...")
            self.cleanup()

            print("Simulation finished.")

    def cleanup(self):
        """Nettoyage des ressources"""
        try:
            if hasattr(self, 'influx_client'):
                self.influx_client.close()
                print("InfluxDB connection closed.")
        except Exception as e:
            print(f"Error closing InfluxDB connection: {e}")

def main():
    parser = argparse.ArgumentParser(description='WNTR Simulation with InfluxDB export for Grafana visualization')
    parser.add_argument('config_file', help='Path to YAML configuration file')

    args = parser.parse_args()

    try:
        simulation = WNTRInfluxSimulation(args.config_file)
        simulation.run_simulation()
    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        print("Program terminated.")

if __name__ == "__main__":
    main()
