class PSO():

    def __init__(self, part_num : int, c1 : float = 2.0, c2 : float = 2.0 , w : float = 1.0, r1 : float = 2.0, r2 : float= 2.0) -> None:
        self.part_num = part_num
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.r1 = r1
        self.r2 = r2

        self.velocity : list[float] = [0 for i in range(part_num)]
        self.pos : list[float] = [0 for i in range(part_num)]

    def set_pos(self, positions : list) -> None:
        if len(positions) != self.part_num:
            raise 