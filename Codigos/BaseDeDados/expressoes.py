import re

certidao_protesto = re.compile(r'((certid[aã]o).{0,100}(protesto))',flags=re.IGNORECASE|re.S)
integralizado = re.compile(r'((capital.{0,20}integralizado)|(patrim[oô]nio.{0,20}integralizado))', flags=re.IGNORECASE|re.S)
idoneidade_financeira = re.compile('((atesta[do]*)|(certid[ãa]o)|(declara[çc]*[ãa]*[o]*))(.{0,70}((i[ni]*doneidade)).{0,20}((financeira)|(banc[áa]ria)))', flags=re.IGNORECASE|re.S)
comprovante_localizacao = re.compile('(((alvar[aá])|(comprov[mnteçcãao]*)).{0,20}((localiza[çc][ãa]o)|(funcionamento)))',flags=re.IGNORECASE|re.S)
n_min_max_limitacao_atestados = re.compile(r'((((dois|duas)|(tr[êe]s)|(quatro)|(cinco)).{0,10}((atestado[s]?)|(certid[aãoões]*))).{0,20}((capacidade t[eé]cnica)|(qualifica[cç][aã]o t[eé]cnica)))', flags=re.IGNORECASE|re.S)
certificado_boas_praticas = re.compile(r'(((certificado[s]?)|(atestado[s]?)|(certid[aã]o)).{0,20}(boa[s]?.{0,5}pr[aá]tica[s]?))', flags=re.IGNORECASE|re.S)
licenca_ambiental = re.compile(r'((licen[cç]a).{0,20}(ambiental))', flags=re.IGNORECASE|re.S)

lista_habilitacao = {'certidao_protesto':certidao_protesto,'certificado_boas_praticas':certificado_boas_praticas, 'integralizado':integralizado, 'idoneidade_financeira':idoneidade_financeira, 
                     'comprovante_localizacao':n_min_max_limitacao_atestados, 'n_min_max_limitacao_atestados':certificado_boas_praticas,
                     'licenca_ambiental':licenca_ambiental, 'comprovante_localizacao':comprovante_localizacao}