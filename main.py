import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, SpanSelector
from matplotlib.animation import FuncAnimation
import sympy as sp
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch, spectrogram
import time
import os

TAUX_ECHANTILLONNAGE = 1000  # Points par seconde
DUREE = 2  # Durée totale du signal en secondes
AMPLITUDE = 1  # Amplitude par défaut
FREQUENCE = 1  # Fréquence par défaut en Hz
NUM_CHANNELS = 2  # Nombre de canaux

# Initialisation des signaux
t = np.linspace(0, DUREE, int(TAUX_ECHANTILLONNAGE * DUREE), endpoint=False)
signals = [np.zeros_like(t) for _ in range(NUM_CHANNELS)]  # Signaux initiaux (vides)

# Création de la figure et des axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.4)

# Axe signal temporel
lines = [ax1.plot(t, signal, label=f'Canal {i+1}')[0] for i, signal in enumerate(signals)]
ax1.set_xlim(0, DUREE)
ax1.set_ylim(-AMPLITUDE * 1.5, AMPLITUDE * 1.5)
ax1.set_xlabel("Temps (s)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Oscilloscope numérique")
ax1.grid(True)
ax1.legend()

# Axe FFT
fft_freq = np.fft.fftfreq(len(t), d=1/TAUX_ECHANTILLONNAGE)
fft_lines = [ax2.plot(fft_freq[:len(t)//2], np.abs(np.fft.fft(signal)[:len(t)//2]), label=f'Canal {i+1}')[0] for i, signal in enumerate(signals)]
ax2.set_xlim(0, 50)
ax2.set_ylim(0, 1000)
ax2.set_xlabel("Fréquence (Hz)")
ax2.set_ylabel("Amplitude FFT")
ax2.set_title("Transformée de Fourier (FFT)")
ax2.grid(True)
ax2.legend()

# Curseurs pour ajuster la fréquence, l'amplitude, le décalage temporel, la phase et le bruit
ax_freq = plt.axes([0.2, 0.25, 0.6, 0.03])
ax_amp = plt.axes([0.2, 0.2, 0.6, 0.03])
ax_shift = plt.axes([0.2, 0.15, 0.6, 0.03])
ax_phase = plt.axes([0.2, 0.1, 0.6, 0.03])
ax_noise = plt.axes([0.2, 0.05, 0.6, 0.03])

freq_slider = Slider(ax_freq, 'Fréquence (Hz)', 0.1, 10.0, valinit=FREQUENCE)
amp_slider = Slider(ax_amp, 'Amplitude', 0.1, 2.0, valinit=AMPLITUDE)
shift_slider = Slider(ax_shift, 'Décalage temporel (s)', -DUREE, DUREE, valinit=0)
phase_slider = Slider(ax_phase, 'Phase (rad)', 0, 2*np.pi, valinit=0)
noise_slider = Slider(ax_noise, 'Bruit', 0, 0.5, valinit=0)

# Boutons pour charger un signal, sauvegarder, exporter et appliquer des filtres
ax_load = plt.axes([0.2, 0.35, 0.2, 0.04])
ax_save = plt.axes([0.4, 0.35, 0.2, 0.04])
ax_export = plt.axes([0.6, 0.35, 0.2, 0.04])
ax_filter = plt.axes([0.2, 0.3, 0.2, 0.04])
ax_peaks = plt.axes([0.4, 0.3, 0.2, 0.04])
ax_power = plt.axes([0.6, 0.3, 0.2, 0.04])
ax_thd = plt.axes([0.8, 0.3, 0.1, 0.04])
ax_spectrogram = plt.axes([0.8, 0.25, 0.1, 0.04])

load_button = Button(ax_load, 'Charger un signal')
save_button = Button(ax_save, 'Sauvegarder le signal')
export_button = Button(ax_export, 'Exporter le graphique')
filter_button = Button(ax_filter, 'Appliquer un filtre')
peaks_button = Button(ax_peaks, 'Détecter les pics')
power_button = Button(ax_power, 'Puissance moyenne')
thd_button = Button(ax_thd, 'Calculer THD')
spectrogram_button = Button(ax_spectrogram, 'Spectrogramme')

# Fonctions prédéfinies
ax_func = plt.axes([0.8, 0.15, 0.1, 0.15])
func_radio = RadioButtons(ax_func, ('sin', 'cos', 'square', 'sawtooth'))

# Fonction pour mettre à jour les signaux
def update_signals(expression):
    global signals
    try:
        # Convertir l'expression en une fonction Python avec sympy
        t_sym = sp.symbols('t')
        expr = sp.sympify(expression)
        func = sp.lambdify(t_sym, expr, modules=['numpy'])
        
        # Génération des signaux avec les paramètres
        for i in range(NUM_CHANNELS):
            signals[i] = func(t + shift_slider.val) * amp_slider.val * np.sin(2 * np.pi * freq_slider.val * t + phase_slider.val)
            signals[i] += noise_slider.val * np.random.normal(0, 1, len(t))
        print("Signaux mis à jour avec succès.")
    except Exception as e:
        print(f"Erreur dans l'expression : {e}")

# Fonction pour animer l'oscilloscope
def animate(frame):
    for i, line in enumerate(lines):
        line.set_ydata(signals[i])  # Mettre à jour les données du signal
    for i, fft_line in enumerate(fft_lines):
        fft_line.set_ydata(np.abs(np.fft.fft(signals[i])[:len(t)//2]))  # Mettre à jour la FFT
    return lines + fft_lines

# Fonction pour charger un signal depuis un fichier
def load_signal_from_file(filename):
    try:
        data = pd.read_csv(filename, header=None)
        global signals
        for i in range(NUM_CHANNELS):
            signals[i] = data.values[:, i][:len(t)]
        print(f"Signaux chargés depuis {filename}.")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")

# Sauvegarder les signaux
def save_signals_to_file(filename):
    try:
        pd.DataFrame(np.array(signals).T).to_csv(filename, index=False, header=False)
        print(f"Signaux sauvegardés dans {filename}.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier : {e}")

# Exporter le graphique
def export_graph(filename):
    try:
        plt.savefig(filename)
        print(f"Graphique exporté dans {filename}.")
    except Exception as e:
        print(f"Erreur lors de l'exportation du graphique : {e}")

# Appliquer un filtre
def apply_filter(cutoff_freq, filter_type='lowpass'):
    nyquist = 0.5 * TAUX_ECHANTILLONNAGE
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(5, normal_cutoff, btype=filter_type, analog=False)
    global signals
    for i in range(NUM_CHANNELS):
        signals[i] = filtfilt(b, a, signals[i])
    print(f"Filtre {filter_type} appliqué avec une fréquence de coupure de {cutoff_freq} Hz.")

# Détecter les pics
def detect_peaks(threshold=0.5):
    for i in range(NUM_CHANNELS):
        peaks, _ = find_peaks(signals[i], height=threshold)
        ax1.plot(t[peaks], signals[i][peaks], "x", color='red', label=f'Pics détectés Canal {i+1}')
    ax1.legend()
    print(f"Pics détectés avec un seuil de {threshold}.")

# Calculer la puissance moyenne
def calculate_power():
    for i in range(NUM_CHANNELS):
        power = np.mean(signals[i]**2)
        print(f"Puissance moyenne du canal {i+1} : {power:.4f}")

# Calculer la distorsion harmonique totale (THD)
def calculate_thd():
    for i in range(NUM_CHANNELS):
        fft_vals = np.fft.fft(signals[i])
        fft_vals = np.abs(fft_vals[:len(t)//2])
        fundamental = np.max(fft_vals)
        harmonic_power = np.sum(fft_vals**2) - fundamental**2
        thd = np.sqrt(harmonic_power) / fundamental
        print(f"Distorsion harmonique totale (THD) du canal {i+1} : {thd:.4f}")

# Calculer la densité spectrale de puissance (PSD)
def calculate_psd():
    for i in range(NUM_CHANNELS):
        freqs, psd = welch(signals[i], fs=TAUX_ECHANTILLONNAGE, nperseg=1024)
        ax2.plot(freqs, psd, label=f'Canal {i+1}')
    ax2.set_xlabel("Fréquence (Hz)")
    ax2.set_ylabel("Densité spectrale de puissance")
    ax2.set_title("Densité spectrale de puissance (PSD)")
    ax2.grid(True)
    ax2.legend()
    plt.draw()

# Afficher un spectrogramme
def show_spectrogram():
    for i in range(NUM_CHANNELS):
        f, t_spec, Sxx = spectrogram(signals[i], fs=TAUX_ECHANTILLONNAGE)
        plt.figure()
        plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.ylabel('Fréquence [Hz]')
        plt.xlabel('Temps [sec]')
        plt.title(f'Spectrogramme du canal {i+1}')
        plt.colorbar(label='Intensité [dB]')
        plt.show()

# Sélectionner une plage de fréquence
def onselect(vmin, vmax):
    global signals
    cutoff_freq = (vmin + vmax) / 2
    apply_filter(cutoff_freq, filter_type='bandpass')
    print(f"Filtre bandpass appliqué avec une plage de {vmin} Hz à {vmax} Hz.")

# Associer les événements aux fonctions
freq_slider.on_changed(lambda val: update_signals(expression_input))
amp_slider.on_changed(lambda val: update_signals(expression_input))
shift_slider.on_changed(lambda val: update_signals(expression_input))
phase_slider.on_changed(lambda val: update_signals(expression_input))
noise_slider.on_changed(lambda val: update_signals(expression_input))

load_button.on_clicked(lambda event: load_signal_from_file(input("Entrez le nom du fichier CSV à charger : ")))
save_button.on_clicked(lambda event: save_signals_to_file(input("Entrez le nom du fichier CSV pour sauvegarder les signaux : ")))
export_button.on_clicked(lambda event: export_graph(input("Entrez le nom du fichier pour exporter le graphique (exemple : graph.png) : ")))
filter_button.on_clicked(lambda event: apply_filter(float(input("Entrez la fréquence de coupure du filtre (Hz) : ")), input("Entrez le type de filtre (lowpass, highpass, bandpass) : ")))
peaks_button.on_clicked(lambda event: detect_peaks(float(input("Entrez le seuil pour la détection des pics : "))))
power_button.on_clicked(lambda event: calculate_power())
thd_button.on_clicked(lambda event: calculate_thd())
spectrogram_button.on_clicked(lambda event: show_spectrogram())

# Oscilloscope
if __name__ == "__main__":
    expression_input = input("Entrez votre fonction mathématique en utilisant 't' comme variable (exemple : sin(2*pi*t)) : ")
    update_signals(expression_input)  # Mettre à jour les signaux
    
    # Démarrer l'animation
    ani = FuncAnimation(fig, animate, frames=range(100), interval=50, blit=True)
    plt.show()