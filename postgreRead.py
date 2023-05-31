import psycopg2
import psycopg2.extras
import pandas as pd


# conn = psycopg2.connect(host="localhost",dbname='postgres',user='postgres',password='123123',port=5432)
# cursor = conn.cursor()
# row = []



def connect_db(host, dbname, user, password, port):
    return psycopg2.connect(host = host,dbname = dbname,user = user,password = password,port = port)
    
def select(conn, dbname):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(f"select * from {dbname}".format(dbname))
    rows = cur.fetchall()
    
    return rows

def dict_maker(header, row):
    result_dict = dict(zip(header,row))
    return result_dict

    # gatway_conn = connect_db(host="localhost",dbname='gateway',user='postgres',password='123123',port=5432)
    
    # cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # cur2 = conn2.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # cur.execute("select * from postgres")
    # cur2.execute("select * from gateway")
    
    # rows = cur.fetchall()

try:

    postgres_data = []
    gateway_data = []
    postgress_conn = connect_db("localhost",'postgres','postgres','123123',5432)
    gateway_conn = connect_db("localhost",'postgres','postgres','123123',5432)
    
    postgres_rows = select(postgress_conn,"postgres")
    
    gateway_rows = select(gateway_conn,"gateway")
    
    postgres_header = ['serviceID','containerID','CPU','Mem','datetime']
    gateway_header = ['service_id','api_id','request_time','response_time']
    
    for row in postgres_rows:
        postgres_data.append(dict_maker(postgres_header,row))
    
    for row in gateway_rows:
        gateway_data.append(dict_maker(gateway_header,row))
    
    final_data = pd.DataFrame(postgres_data)
    
    for i in range(len(gateway_data)):
        for k, v in gateway_data[i].items():
            if k == 'service_id':
                pass

            else:
                final_data[k] = v
                
        for index, row in final_data.iterrows():
        # if final_data[i]['api_id'] == final_data[i]['container_id']:
        #     print(final_data[i])
        
            if row['containerID'] == row['api_id']:
                print(row)
    
    # gateway_data.append(dict_maker(header,gateway_rows))
    # lst = [i[0].split(',') for i in header]
    # df = []
    # resource_dict = {}
    
    # for row in rows:
    #     result_dict = dict(zip(header,row))
    #     df.append(result_dict)
        
        # df.append(dict)
    # print([dict(zip(lst[0], v)) for v in lst[1:]])
    
    # print(postgres_data)
    # print(gateway_data)
except psycopg2.DatabaseError as db_err:
    print(db_err)
    


# def get_dict_resultset(sql):
#     conn = psycopg2.connect(host="localhost",dbname='postgres',user='postgres',password='123123',port=5432)
#     cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
#     cur.execute (sql)
#     ans =cur.fetchall()
#     dict_result = []
#     for row in ans:
#         dict_result.append(dict(row))
#     return dict_result

# sql = "select * from postgres"

