"""
Dependencies: pygame (1.9.6)
"""

class DS2_Controller:
    def __init__(self, ds2_joystick, L=None, R=None, hat=None, triangle=None, square=None, cross=None, circle=None, start=None,
                 select=None, analog_left_x=None, analog_left_y=None, analog_right_x=None, analog_right_y=None):
        self.ds2_joystick = ds2_joystick
        self.L = [0, 0] if L is None else L  # corresponds to L1, L2 respectively
        self.R = [0, 0] if R is None else R  # corresponds to R1, R2 respectively
        self.hat = (0, 0) if hat is None else hat  # corresponds to direction buttons on the left

        # symbol buttons on the right
        self.triangle = False if triangle is None else triangle
        self.square = False if square is None else square
        self.cross = False if cross is None else cross
        self.circle = False if circle is None else circle

        self.start = False if start is None else start
        self.select = False if select is None else select

        # analog sticks
        self.analog_left_x = 0 if analog_left_x is None else analog_left_x
        self.analog_left_y = 0 if analog_left_y is None else analog_left_y
        self.analog_right_x = 0 if analog_right_x is None else analog_right_x
        self.analog_right_y = 0 if analog_right_y is None else analog_right_y

    """ Analog """
    def update_analog(self):
        self.analog_left_x = self.ds2_joystick.get_axis(0)
        self.analog_left_y = self.ds2_joystick.get_axis(1)
        self.analog_right_x = self.ds2_joystick.get_axis(3)
        self.analog_right_y = self.ds2_joystick.get_axis(2)

    """ Buttons """
    def update_buttons(self):
        # symbols on the right
        self.triangle = self.ds2_joystick.get_button(0)
        self.circle = self.ds2_joystick.get_button(1)
        self.cross = self.ds2_joystick.get_button(2)
        self.square = self.ds2_joystick.get_button(3)

        # L1, L2 and R1, R2 respectively
        self.L = [self.ds2_joystick.get_button(6), self.ds2_joystick.get_button(4)]
        self.R = [self.ds2_joystick.get_button(7), self.ds2_joystick.get_button(5)]

        # select and start
        self.select = self.ds2_joystick.get_button(8)
        self.start = self.ds2_joystick.get_button(9)

        # hat (direction buttons on the left)
        self.hat = self.ds2_joystick.get_hat(0)

    def update(self):
        """:self
            update all buttons and analogs on the controller
        """
        self.update_buttons()
        self.update_analog()

    def debug(self):
        print(
            f"hat: {self.hat}   "
            f"analog left: {round(self.analog_left_x, 3)}, {round(self.analog_left_y, 3)}   "
            f"analog right: {round(self.analog_right_x, 3)}, {round(self.analog_right_y, 3)}   "
            f"triangle: {self.triangle}   circle: {self.circle}   cross: {self.cross}   square: {self.square}   "
            f"L: {self.L}   R: {self.R}"
        )





