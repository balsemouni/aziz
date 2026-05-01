# # un AGC (Automatic Gain Control), soit un Contrôle Automatique du Gain.
# # Prevents sound from being too loud or too quiet.
# import numpy as np

# class SimpleAGC:
#     """
#     Lightweight AGC for speech
#     """
#     def __init__(self, target_rms=0.02, max_gain=10.0):
#         #RMS (Root Mean Square) qui est une mesure de la puissance moyenne du son.
#         self.target_rms = target_rms
#         #C'est une sécurité. On ne veut pas multiplier le son par plus de 10,
#         # car sinon on amplifierait aussi énormément le bruit de fond (souffle) quand il n'y a pas de parole.
#         self.max_gain = max_gain
#     #Calcul de l'énergie actuelle
#     def process(self, audio):
#         rms = np.sqrt(np.mean(audio**2) + 1e-8)
#         if rms < 1e-5:
#             return audio

#         gain = min(self.target_rms / rms, self.max_gain)
#         return audio * gain

import numpy as np

class SimpleAGC:
    """
    Zero-latency AGC for speech
    """
    def __init__(self, target_rms=0.02, max_gain=10.0, attack_ms=5, release_ms=20):
        self.target_rms = target_rms
        self.max_gain = max_gain
        self.current_gain = 1.0
        
        # Time constants for smooth gain changes
        self.sample_rate = 16000
        self.attack_coeff = np.exp(-1.0 / (attack_ms * self.sample_rate / 1000))
        self.release_coeff = np.exp(-1.0 / (release_ms * self.sample_rate / 1000))
    
    def process(self, audio):
        """Process audio with smooth gain adjustment"""
        if len(audio) == 0:
            return audio
        
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2) + 1e-8)
        
        if rms < 1e-5:
            return audio
        
        # Desired gain
        desired_gain = min(self.target_rms / rms, self.max_gain)
        
        # Smooth gain transition
        if desired_gain > self.current_gain:
            # Attack (increase gain quickly)
            coeff = self.attack_coeff
        else:
            # Release (decrease gain slowly)
            coeff = self.release_coeff
        
        self.current_gain = coeff * self.current_gain + (1 - coeff) * desired_gain
        
        # Apply gain
        return audio * self.current_gain
    
    def reset(self):
        """Reset AGC state"""
        self.current_gain = 1.0