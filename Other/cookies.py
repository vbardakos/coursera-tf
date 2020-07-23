class CookieRequests(object):

    def __init__(self, my_cookies, my_packs, my_boxes):
        self.c = my_cookies
        self.p = my_packs
        self.b = my_boxes
        self.unit_list = [my_boxes * my_packs, my_packs]

    @classmethod
    def ask_for_cookies(cls, question: str = "",
                        per_pack: int = 10, per_box: int = 10, patience: int = 5):
        assert isinstance((per_pack and per_box), int) and (per_pack and per_box) > 1
        cookies = 0
        if not question:
            question = 'Enter how many packs of cookies you want to buy: '

        for _ in range(patience):
            str_cookies = input(question)
            try:
                if int(str_cookies) > 0:
                    cookies = int(str_cookies)
                    break
                else:
                    print('No cookies requested. Next...')
            except ValueError:
                print(f'Warning : You cannot ask for {str_cookies}! Next...')
        if cookies:
            return cls(cookies, per_pack, per_box)

    @property
    def get_units(self):
        box, pack, cookie = self.__box_pack_list__(self.c, iter(self.unit_list))
        return f"cookies are distributed to {box} boxes, {pack} packs and {cookie} remaining"

    def __box_pack_list__(self, cookies: int, cookie_iterator):
        for i in cookie_iterator:
            divisor, remainder = divmod(cookies, i)
            return [divisor] + self.__box_pack_list__(remainder, cookie_iterator)
        return [cookies]


class CookieShop(CookieRequests):

    def __init__(self, num_cookies: int = 0, packs=0, boxes=0, patience=1):
        if num_cookies and packs and boxes:
            super(CookieShop, self).__init__(num_cookies, packs, boxes)

    @classmethod
    def ask_for_sales(cls, patience: int = 5):
        cls.sales = 0
        for _ in range(patience):
            sales = input('What\'s the price of your cookies? ')
            try:
                if float(sales) > 0:
                    cls.sales = round(float(sales), 2)
                    break
            except ValueError:
                print('This is not a number')

    def compute_profits(self):
        box, pack, cookies = self.__box_pack_list__(self.c, iter(self.unit_list))
        total = self.sales * (self.p * (self.b * box * self.sales + pack) + cookies)
        return round(total, 2)

    def commissions(self):
        pass

    @property
    def get_profits(self):
        return f"After selling {self.c} cookies @ ${self.sales}, the total profit is {self.compute_profits()}"





def recursive_2(num, my_iter):
    for i in my_iter:
        d, r = divmod(num, i)
        return [d] + recursive_2(r, my_iter)
    return [num]


if __name__ == '__main__':

    # shopping = CookieRequests.ask_for_cookies(per_box=100, per_pack=10, patience=2)
    # answer = shopping.cookie_units
    # print(answer)

    CookieShop.ask_for_cookies(per_pack=10, per_box=10, patience=2)
    CookieShop.ask_for_sales(patience=3)