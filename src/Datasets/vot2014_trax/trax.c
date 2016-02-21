/* -*- Mode: C; indent-tabs-mode: nil; c-basic-offset: 4; tab-width: 4 -*- */

//#include <unistd.h>
#include <string.h>
#include <assert.h>

#include "trax.h"
#include "region.h"
#include "strmap.h"
#include "buffer.h"
#include "message.h"

#define VALIDATE_HANDLE(H) assert((H->flags & TRAX_FLAG_VALID))
#define VALIDATE_ALIVE_HANDLE(H) assert((H->flags & TRAX_FLAG_VALID) && !(H->flags & TRAX_FLAG_TERMINATED))
#define VALIDATE_SERVER_HANDLE(H) assert((H->flags & TRAX_FLAG_VALID) && (H->flags & TRAX_FLAG_SERVER))
#define VALIDATE_CLIENT_HANDLE(H) assert((H->flags & TRAX_FLAG_VALID) && !(H->flags & TRAX_FLAG_SERVER))

#define BUFFER_LENGTH 64

#define REGION(VP) ((Region*) (VP))

#define REGION_TYPE(VP) ((((Region*) (VP))->type == RECTANGLE) ? TRAX_REGION_RECTANGLE : (((Region*) (VP))->type == POLYGON ? TRAX_REGION_POLYGON : TRAX_REGION_SPECIAL))

#define REGION_TYPE_BACK(T) (( T == TRAX_REGION_RECTANGLE ) ? RECTANGLE : ( T == TRAX_REGION_POLYGON ? POLYGON : SPECIAL))

struct trax_properties {
    StrMap *map;
};

void copy_property(const char *key, const char *value, const void *obj) {

    trax_properties* dest = (trax_properties*) obj;

    trax_properties_set(dest, key, value);

}


void copy_properties(trax_properties* source, trax_properties* dest) {
    trax_properties_enumerate(source, copy_property, dest);

}

trax_handle* trax_client_setup(FILE* input, FILE* output, FILE* log) {

    trax_properties* tmp_properties;
    string_list arguments;
    int version = 1;
    char* tmp;

    trax_handle* client = (trax_handle*) malloc(sizeof(trax_handle));

    client->flags = (0 & ~TRAX_FLAG_SERVER) | TRAX_FLAG_VALID;

    client->log = log;
    client->input = input;
    client->output = output;

    tmp_properties = trax_properties_create();
    LIST_CREATE(arguments, 8);
    
    if (read_message(client->input, client->log, &arguments, tmp_properties) != TRAX_HELLO) {
        goto failure;
    }

    if (LIST_SIZE(arguments) > 0)
        goto failure;

    tmp = trax_properties_get(tmp_properties, "trax.region");
    client->config.format_region = TRAX_REGION_RECTANGLE;
    if (tmp && strcmp(tmp, "polygon") == 0)
        client->config.format_region = TRAX_REGION_POLYGON;
    free(tmp);

  
        
    client->config.format_image = TRAX_IMAGE_PATH;
    client->version = trax_properties_get_int(tmp_properties, "trax.version", 1);

    trax_properties_release(&tmp_properties);
    LIST_DESTROY(arguments);

    return client;
    
failure:

    LIST_DESTROY(arguments);
    trax_properties_release(&tmp_properties);
    free(client);
    return NULL;

}

int trax_client_wait(trax_handle* client, trax_region** region, trax_properties* properties) {

    trax_properties* tmp_properties;
    string_list arguments;
    int result = TRAX_ERROR;

    (*region) = NULL;

    VALIDATE_ALIVE_HANDLE(client);
    VALIDATE_CLIENT_HANDLE(client);

    tmp_properties = trax_properties_create();
    LIST_CREATE(arguments, 8);

    result = read_message(client->input, client->log, &arguments, tmp_properties);

    if (result == TRAX_STATUS) {

        if (LIST_SIZE(arguments) != 1)
            goto failure;

        Region* _region = NULL;

        result = TRAX_STATUS;

        if (!region_parse(arguments.buffer[0], &_region)) {
            goto failure;
        }  
          
        (*region) = _region;

        copy_properties(tmp_properties, properties);

        goto end;

    } else if (result = TRAX_QUIT) {

        if (LIST_SIZE(arguments) != 0)
            goto failure;

        if (properties) 
            copy_properties(tmp_properties, properties);

        client->flags |= TRAX_FLAG_TERMINATED;

        goto end;
    } 

failure:
    result = TRAX_ERROR;

end:

    LIST_DESTROY(arguments);
    trax_properties_release(&tmp_properties);

    return result;

}

void trax_client_initialize(trax_handle* client, trax_image* image, trax_region* region, trax_properties* properties) {

	char* data = NULL;
	Region* _region;
    string_list arguments;

    VALIDATE_ALIVE_HANDLE(client);
    VALIDATE_CLIENT_HANDLE(client);

	assert(image && region);

    _region = REGION(region);

    assert(_region->type != SPECIAL);

    LIST_CREATE(arguments, 2);

    if (image->type == TRAX_IMAGE_PATH) {
        LIST_APPEND(arguments, image->data);
    } else goto failure;

    if (client->config.format_region != REGION_TYPE(region)) {

        trax_region* converted = region_convert(region, client->config.format_region);

        assert(converted);

        data = region_string(converted);

        trax_region_release(&converted);

    } else data = region_string(region);

    if (data) {
        LIST_APPEND(arguments, data);
        free(data);
    }

    write_message(client->output, client->log, TRAX_INITIALIZE, arguments, properties);

failure:

    LIST_DESTROY(arguments);

}

void trax_client_frame(trax_handle* client, trax_image* image, trax_properties* properties) {

    VALIDATE_ALIVE_HANDLE(client);
    VALIDATE_CLIENT_HANDLE(client);
    string_list arguments;

    assert(client->config.format_image == image->type);

    LIST_CREATE(arguments, 2);

    if (image->type == TRAX_IMAGE_PATH) {
        LIST_APPEND(arguments, image->data);
    } else goto failure;

    write_message(client->output, client->log, TRAX_FRAME, arguments, properties);

failure:

    LIST_DESTROY(arguments);
}

trax_handle* trax_server_setup_standard(trax_configuration config, FILE* log) {

    return trax_server_setup(config, stdin, stdout, log);

}

trax_handle* trax_server_setup(trax_configuration config, FILE* input, FILE* output, FILE* log) {

    trax_properties* properties;
    trax_handle* server = (trax_handle*) malloc(sizeof(trax_handle));
    string_list arguments;

    server->flags = (TRAX_FLAG_SERVER) | TRAX_FLAG_VALID;

    server->log = log;
    server->input = input;
    server->output = output;

    properties = trax_properties_create();

    trax_properties_set_int(properties, "trax.version", TRAX_VERSION);

    switch (config.format_region) {
        case TRAX_REGION_RECTANGLE:
            trax_properties_set(properties, "trax.region", "rectangle");
            break;
        case TRAX_REGION_POLYGON:
            trax_properties_set(properties, "trax.region", "polygon");
            break;
        default:
            config.format_region = TRAX_REGION_RECTANGLE;
            trax_properties_set(properties, "trax.region", "rectangle");
            break;
    }

    switch (config.format_image) {
        case TRAX_IMAGE_PATH:
            trax_properties_set(properties, "trax.image", "path");
            break;
        default:
            config.format_image = TRAX_IMAGE_PATH;
            trax_properties_set(properties, "trax.image", "path");
            break;
    }

    server->config = config;
   
    LIST_CREATE(arguments, 1);

    write_message(server->output, server->log, TRAX_HELLO, arguments, properties);

    trax_properties_release(&properties);

    LIST_DESTROY(arguments);

    return server;

}

int trax_server_wait(trax_handle* server, trax_image** image, trax_region** region, trax_properties* properties) 
{

    int result = TRAX_ERROR;
    string_list arguments;
    trax_properties* tmp_properties;

    VALIDATE_ALIVE_HANDLE(server);
    VALIDATE_SERVER_HANDLE(server);

    tmp_properties = trax_properties_create();
    LIST_CREATE(arguments, 8);

    result = read_message(server->input, server->log, &arguments, tmp_properties);

    if (result == TRAX_FRAME) {

        if (LIST_SIZE(arguments) != 1)
            goto failure;

        switch (server->config.format_image) {
        case TRAX_IMAGE_PATH: {
            *image = trax_image_create_path(arguments.buffer[0]);
            break;
        }
        default:
            goto failure;
        }

        if (properties) 
            copy_properties(tmp_properties, properties);  

        goto end;
    } else if (result == TRAX_QUIT) {

        if (LIST_SIZE(arguments) != 0)
            goto failure;

        if (properties) 
            copy_properties(tmp_properties, properties);  

        server->flags |= TRAX_FLAG_TERMINATED;

        goto end;
    } else if (result == TRAX_INITIALIZE) {

        if (LIST_SIZE(arguments) != 2)
            goto failure;

        switch (server->config.format_image) {
        case TRAX_IMAGE_PATH: {
            *image = trax_image_create_path(arguments.buffer[0]);
            break;
        }
        default:
            goto failure;
        }

        if (!region_parse(arguments.buffer[1], (Region**)region)) {
            goto failure;
        }

        if (properties) 
            copy_properties(tmp_properties, properties);  

        goto end;
    }

failure:

    result = TRAX_ERROR;

end:

    LIST_DESTROY(arguments);
    trax_properties_release(&tmp_properties);

    return result;
}

void trax_server_reply(trax_handle* server, trax_region* region, trax_properties* properties) {

	char* data;
    string_list arguments;

    VALIDATE_ALIVE_HANDLE(server);
    VALIDATE_SERVER_HANDLE(server);

    data = region_string(REGION(region));

    if (!data) return;

    LIST_CREATE(arguments, 1);

    if (data) {
        LIST_APPEND(arguments, data);
        free(data);
    }

    write_message(server->output, server->log, TRAX_STATUS, arguments, properties);

    LIST_DESTROY(arguments);

}

int trax_cleanup(trax_handle** handle) {

    if (!*handle) return -1;

    VALIDATE_HANDLE((*handle));

    if (!((*handle)->flags & TRAX_FLAG_TERMINATED)) {
        string_list arguments;
        trax_properties* tmp_properties;

        LIST_CREATE(arguments, 1);
        tmp_properties = trax_properties_create();

        write_message((*handle)->output, (*handle)->log, TRAX_QUIT, arguments, tmp_properties);

        LIST_DESTROY(arguments);
        trax_properties_release(&tmp_properties);

    }

    (*handle)->flags |= TRAX_FLAG_TERMINATED;

    if ((*handle)->log) {
        (*handle)->log = 0;
    }

    free(*handle);
    *handle = 0;

    return 0;
}

void trax_image_release(trax_image** image) {

    switch ((*image)->type) {
        case TRAX_IMAGE_PATH:
            free((*image)->data);
    }

    free(*image);

    *image = NULL;
}

trax_image* trax_image_create_path(const char* path) {

    trax_image* img = (trax_image*) malloc(sizeof(trax_image));

    img->type = TRAX_IMAGE_PATH;
    img->width = 0;
    img->height = 0;
    img->data = (char*) malloc(sizeof(char) * (strlen(path) + 1));
    strcpy(img->data, path);

    return img;
}

const char* trax_image_get_path(trax_image* image) {

    assert(image->type == TRAX_IMAGE_PATH);

    return image->data;

}

void trax_region_release(trax_region** region) {

    Region* _region = REGION(*region);

    region_release(&_region);

    *region = NULL;

}

trax_region* trax_region_create_special(int code) {

    return region_create_special(code);

}

trax_region* trax_region_create_polygon(int size) {

    assert(size > 2);
   
    return region_create_polygon(size);

}

trax_region* trax_region_create_rectangle(float x, float y, float width, float height) {

    return region_create_rectangle(x, y, width, height);

}

trax_region* trax_region_get_bounds(const trax_region* region) {

    return region_convert(REGION(region), RECTANGLE);

}

int trax_region_get_type(const trax_region* region) {

    return REGION_TYPE(region);

}

void trax_region_set_special(trax_region* region, int code) {

    assert(REGION(region)->type == SPECIAL);

    REGION(region)->data.special = (int) code;

}


int trax_region_get_special(const trax_region* region) {

    assert(REGION(region)->type == SPECIAL);

    return REGION(region)->data.special;

}

void trax_region_set_rectangle(trax_region* region, float x, float y, float width, float height) {

    assert(REGION(region)->type == RECTANGLE);

    REGION(region)->data.rectangle.x = x;
    REGION(region)->data.rectangle.y = y;
    REGION(region)->data.rectangle.width = width;
    REGION(region)->data.rectangle.height = height;

}

void trax_region_get_rectangle(const trax_region* region, float* x, float* y, float* width, float* height) {

    assert(REGION(region)->type == RECTANGLE);

    *x = REGION(region)->data.rectangle.x;
    *y = REGION(region)->data.rectangle.y;
    *width = REGION(region)->data.rectangle.width;
    *height = REGION(region)->data.rectangle.height;

}

void trax_region_set_polygon_point(trax_region* region, int index, float x, float y) {

    assert(REGION(region)->type == POLYGON);

    assert(index >= 0 || index < (REGION(region)->data.polygon.count));

    REGION(region)->data.polygon.x[index] = x;
    REGION(region)->data.polygon.y[index] = y;
}

void trax_region_get_polygon_point(const trax_region* region, int index, float* x, float* y) {

    assert(REGION(region)->type == POLYGON);

    assert (index >= 0 || index < (REGION(region)->data.polygon.count));

    *x = REGION(region)->data.polygon.x[index];
    *y = REGION(region)->data.polygon.y[index];
}

int trax_region_get_polygon_count(const trax_region* region) {

    assert(REGION(region)->type == POLYGON);

    return REGION(region)->data.polygon.count;

}

void trax_properties_release(trax_properties** properties) {
    
    if (properties && *properties) {
        if ((*properties)->map) sm_delete((*properties)->map);
        free((*properties));
        *properties = 0;
    }

}

void trax_properties_clear(trax_properties* properties) {
    
    if (properties) {
        if (properties->map) 
            sm_delete(properties->map);
        properties->map = sm_new(32);
    }
}

trax_properties* trax_properties_create() {

    trax_properties* prop = malloc(sizeof(trax_properties));

    prop->map = sm_new(32);

    return prop;

}

void trax_properties_set(trax_properties* properties, const char* key, const char* value) {

    sm_put(properties->map, key, value);

}

void trax_properties_set_int(trax_properties* properties, const char* key, int value) {

    char tmp[128];
    sprintf(tmp, "%d", value);
    trax_properties_set(properties, key, tmp);

}

void trax_properties_set_float(trax_properties* properties, const char* key, float value) {

    char tmp[128];
    sprintf(tmp, "%f", value);
    trax_properties_set(properties, key, tmp);

}

char* trax_properties_get(const trax_properties* properties, const char* key) {

	char* value;
    int size = sm_get(properties->map, key, NULL, 0);

    if (size < 1) return NULL;

    value = (char *) malloc(sizeof(trax_properties) * size);

    sm_get(properties->map, key, value, size);

    return value;
}

int trax_properties_get_int(const trax_properties* properties, const char* key, int def) {

    char* end;
    long ret;
    char* value = trax_properties_get(properties, key);

    if (value == NULL) return def;

    if (value[0]!='\0') {
        ret = strtol(value, &end, 10);
        ret = (*end=='\0' && end!=value) ? ret : def;
    }

    free(value);
    return (int)ret;

}


float trax_properties_get_float(const trax_properties* properties, const char* key, float def) {

    char* end;
    float ret;
    char* value = trax_properties_get(properties, key);

    if (value == NULL) return def;

    if (value[0]!='\0') {
        ret = strtod(value,&end);
        ret = (*end=='\0' && end!=value) ? ret : def;
    }

    free(value);
    return ret;

}

void trax_properties_enumerate(trax_properties* properties, trax_enumerator enumerator, void* object) {
    if (properties && enumerator) {
        
        sm_enum(properties->map, enumerator, object);
    }
}

