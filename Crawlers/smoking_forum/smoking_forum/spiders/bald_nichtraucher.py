
import scrapy

class nichtraucherSpiderClass(scrapy.Spider):

    name = "bald_nichtraucher"

    # https://stackoverflow.com/questions/28410071/start-urls-in-scrapy
    start_urls = ['https://www.endlich-nichtraucher-forum.de/forums/nichtraucher-bald-bin-ich-soweit.11/page-1']

    def parse(self, response):
        page = 1
        while page <= 3:
            url = "https://www.endlich-nichtraucher-forum.de/forums/nichtraucher-bald-bin-ich-soweit.11/page-"+str(page)
            page += 1
            yield response.follow(url, self.parse_page)

    def parse_page(self, response):

        threads = response.xpath("//*[@class = 'structItem-title']")

        for thread in threads:  
            thread_absolute_link = thread.css('a::attr(href)').get()
            yield response.follow(thread_absolute_link, self.parse_thread)


    def parse_thread(self, response):

        with open('post_ids.txt') as f:
            post_ids = [ line.strip() for line in f ]

        comments = response.xpath("//*[@class = 'block-body js-replyNewMessageContainer']/article")

        title = response.xpath("//*[@class = 'p-title-value']/text()").extract_first()
        thread_id = response.css('html::attr(data-content-key)').get()

        for comment in comments:

            post_id = comment.css('article::attr(data-content)').get()

            if (post_id in post_ids):

                content = "".join(comment.xpath(".//*[@class = 'bbWrapper']/text()").extract())
                content = content.replace("\t", "")
                content = content.replace("\r", "")
                    #post_id = "".join(comment.xpath(".//*[@class = 'messageText']/text()").extract())
                username = "".join(comment.css("article::attr(data-author)").get())
                date = "".join(comment.css("time::attr(datetime)").get())

                yield {
                    "title" : title,
                    "thread_id": thread_id,
                    "post_id": post_id,
                    "date" : date,
                    "username" : username,
                    "content" : content,
                }

        next_page = response.xpath("//*[@class='pageNav-jump pageNav-jump--next']/@href").extract_first()

        if (next_page is not None):
            yield response.follow(next_page, self.parse_thread)
