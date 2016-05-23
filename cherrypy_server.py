import evospace
import cherrypy

from cherrypy._cpcompat import ntou

import json, os


class Content():
    def __init__(self, popName="pop"):
        self.population = evospace.Population(popName)
        self.population.initialize()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in(content_type=[ntou('application/json'),
                                          ntou('text/javascript'),
                                          ntou('application/json-rpc')
                                          ])
    def evospace(self):
        if cherrypy.request.json:
            obj = cherrypy.request.json
            method = obj["method"]
            _id = obj["id"]

            if "params" in obj:
                params = obj["params"]
            else:
                return json.dumps({"result": None, "error":
                    {"code": -32604, "message": "Params empty"}, "id": _id})

            # process the data
            cherrypy.response.headers['Content-Type'] = 'text/json-comment-filtered'
            result = None
            if method == "initialize":
                result = self.population.initialize()
                return json.dumps({"result": result, "error": None, "id": _id})
            #
            # if method == "getSample":
            #     result = self.population.get_sample(params[0])
            #     if result:
            #         return json.dumps({"result": result, "error": None, "id": _id})
            #     else:
            #         return json.dumps({"result": None, "error":
            #             {"code": -32601, "message": "EvoSpace empty"}, "id": _id})
            # elif method == "respawn":
            #     result = self.population.respawn(params[0])
            # elif method == "putSample":
            #     result = self.population.put_sample(params[0])
            elif method == "putIndividual":
                result = self.population.put_individual(**params[0])
            elif method == "size":
                result = self.population.size()
            # elif method == "found":
            #     result = self.population.found()
            # elif method == "found_it":
            #     result = self.population.found_it()
            if method == "mensaje":
                return json.dumps([1, 2, 3, {'4': 5, '6': 7}], separators=(',', ':'))

            return json.dumps({"result": result, "error": None, "id": _id})

        else:
            print "blah"
            return "blah"

    # site_header = """<html>
    #       <head>
    #         <link href="/static/css/style.css" rel="stylesheet">
    #       </head>
    #       <body>
    #  """
    #
    #
    # site_end = """  </body>
    #              </html>"""
    #
    #
    # @cherrypy.expose
    # def index(self):
    #     site_body = """      <form method="get" action="evospace">
    #                         <input type="text" value="8" name="length" />
    #                         <button type="submit">Give it now!</button>
    #                       </form>"""
    #
    #     result = (self.site_header, site_body, self.site_end)
    #
    #     return result
    # @cherrypy.expose
    # def index(self):
    #     return "Servidor Funcionando"


config={
    'global':{
        'server.socket_host' : '0.0.0.0',
        'server.socket_port' : int(os.environ.get('PORT', '5060')),
        'server.thread_pool' : 200,
        'server.socket_timeout': 30
    },
    '/':{
        'tools.sessions.on': False
    }
}
def error_page_404(status, message, traceback, version):
    return "404 Error!"
def start_server():
    from cherrypy.process import servers

    def fake_wait_for_occupied_port(host, port):
        return

    servers.wait_for_occupied_port = fake_wait_for_occupied_port

    cherrypy.tree.mount(Content(), '/')
    cherrypy.config.update({'error_page.404': error_page_404})
    cherrypy.config.update({'server.socket_port': 5060})
    cherrypy.engine.start()

if __name__ == '__main__':
    # cherrypy.config.update({'server.socket_host': '0.0.0.0',
    #                         'server.socket_port': int(os.environ.get('PORT', '5060'))
    #                            , 'server.environment': 'production'
    #                            , 'server.thread_pool': 200
    #                            , 'tools.sessions.on': False
    #                            , 'server.socket_timeout': 30
    #                         })

    start_server()

    #cherrypy.quickstart(Content('pop'), '/evospace', config)