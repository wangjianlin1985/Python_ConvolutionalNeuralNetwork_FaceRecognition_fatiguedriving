
import time
import pygame
pygame.mixer.init()
print("播放音乐1")
track = pygame.mixer.music.load('1.mp3')
pygame.mixer.music.play()
time.sleep(10)
pygame.mixer.music.stop()
print('结束')