import pygame, sys, random

speed=15

frame_size_x=720
frame_size_y=488

chek_errors=pygame.init();

if(chek_errors[1]>0):

    print("HATA"+chek_errors[1])
    
else:
    print("OYUN BASLADI")


    pygame.display.set_caption("yilan oyunu")
    game_windows=pygame.display.set_mode((frame_size_x , frame_size_y))

    black=pygame.Color(0,0,0)
    white=pygame.Color(255,255,255)
    red=pygame.Color(255,0,0)
    green=pygame.Color(0,255,0)
    blue=pygame.Color(0,0,255)



    fps_controler=pygame.time.Clock()

    square_size=20

    def init_vars():
        global head_pos , snake_body , food_pos , food_spawn , score, direction
        direction="SAG"
        head_pos=[120,60]
        snake_body=[[120,60]]
        food_pos=[random.randrange(1,(frame_size_x//square_size))*square_size,
                random.randrange(1,(frame_size_y//square_size))*square_size]
        food_spawn=True
        score=0


    init_vars()
    def show_score(choice, clor, font, size):
        score_font= pygame.font.SysFont(font, size)
        score_surface= score_font.render("skor: "+ str(score), True, clor)
        score_rect = score_surface.get_rect()
        if choice == 1:
            score_rect.midtop = (frame_size_x / 10, 15)
        else:
            score_rect.midtop = (frame_size_x/2, frame_size_y/1.25)

            game_windows.blit(score_surface, score_rect)

    while True:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type==pygame.KEYDOWN:
                if (event.key == pygame.K_UP or event.key == ord("W")) and direction !="ASSAGI":
                    direction="YUKARI"
                elif(event.key == pygame.K_DOWN or event.key == ord("S")) and direction !="YUKARI":
                    direction="ASSAGI"
                elif(event.key == pygame.K_LEFT or event.key == ord("A")) and direction !="SAG":
                    direction="SOL"
                elif(event.key == pygame.K_RIGHT or event.key == ord("D")) and direction !="SOL":
                    direction="SAG"


        if direction == "YUKARI":
            head_pos[1]-= square_size
        elif direction =="ASSAGI":
            head_pos[1]+=square_size
        elif direction =="SOL":
            head_pos[0]-=square_size
        elif direction == "SAG":
            head_pos[0]+=square_size

        if head_pos[0] < 0:
            head_pos[0] = 0
        elif head_pos[0] > frame_size_x - square_size:
            head_pos[0] = 0
        elif head_pos[1] < 0:
            head_pos[1] = 0
        elif head_pos[1] > frame_size_y - square_size:
            head_pos[1] = 0


        snake_body.insert(0, list(head_pos))
        if head_pos[0] == food_pos[0] and head_pos[1] == food_pos[1]:
            score+=1
            food_spawn=False
        else:
            snake_body.pop()



        if not food_spawn:
            food_pos=[random.randrange(1,(frame_size_x//square_size))*square_size,
                random.randrange(1,(frame_size_y//square_size))*square_size]
            food_spawn = True 


        game_windows.fill(black)
        for pos in snake_body:
            rectVal = pygame.Rect(pos[0] + 2, pos[1] + 2, square_size-2, square_size-2)
            pygame.draw.rect(game_windows, blue, rectVal)

        pygame.draw.rect(game_windows, red, pygame.Rect(food_pos[0], food_pos[1], square_size, square_size)) 


        for block in snake_body[1:]:
            if head_pos[0] == block[0] and head_pos[1] == block[1]:
                init_vars()

        show_score(1,white, 'consolas', 20)
        pygame.display.update()
        fps_controler.tick(speed)