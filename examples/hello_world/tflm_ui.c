#include "rtthread.h"
#include "bf0_hal.h"
#include "drv_io.h"
#include "littlevgl2rtt.h"
#include "string.h"

#include "bf0_pm.h"
#include "drv_gpio.h"
#include "lv_timer.h"
#include "lv_obj_pos.h"
#define LCD_DEVICE_NAME  "lcd"
#define TOUCH_NAME  "touch"
static lv_obj_t *global_label1;
static lv_obj_t *global_label2;
static lv_obj_t *global_canvas;
#define CANVAS_WIDTH 320
#define CANVAS_HEIGHT 300
static lv_color_t global_canvas_buf[CANVAS_WIDTH*CANVAS_HEIGHT];

rt_err_t tflm_ui_obj_init(void)
{
    lv_obj_t * scr = lv_screen_active();
    global_label1 = lv_label_create(scr);//top text

    lv_label_set_long_mode(global_label1, LV_LABEL_LONG_WRAP);
    //lv_obj_set_width(global_label1, LV_HOR_RES_MAX);
    lv_obj_set_style_text_align(global_label1, LV_TEXT_ALIGN_CENTER, 0);
    lv_label_set_text(global_label1, "TFLM sin(x) demo");
    lv_obj_align_to(global_label1, scr, LV_ALIGN_TOP_MID, 0, 20);

    global_label2 = lv_label_create(scr);//output text

    lv_label_set_long_mode(global_label2, LV_LABEL_LONG_WRAP);  /*Break the long lines*/
    //lv_obj_set_width(global_label2, LV_HOR_RES_MAX);
    lv_obj_set_style_text_align(global_label2, LV_TEXT_ALIGN_LEFT, 0);
    lv_label_set_text(global_label2, "output log");
    lv_obj_align_to(global_label2, scr, LV_ALIGN_BOTTOM_MID, 0, -50);

    global_canvas = lv_canvas_create(scr);
    lv_obj_set_size(global_canvas, CANVAS_WIDTH, CANVAS_HEIGHT);
    lv_obj_align_to(global_canvas, scr, LV_ALIGN_CENTER, 0, 0);
    lv_canvas_fill_bg(global_canvas, lv_color_hex(0x000000), LV_OPA_TRANSP);
    lv_canvas_set_buffer(global_canvas, global_canvas_buf, CANVAS_WIDTH, CANVAS_HEIGHT, LV_COLOR_FORMAT_RGB565);
    for (int i = 0; i < CANVAS_WIDTH; i++)
    {
        lv_canvas_set_px(global_canvas, i, 150, lv_color_hex(0xFFFFFF), LV_OPA_COVER);
    }

    return RT_EOK;
}

void tflm_ui_refresh(float x, float y, float y_pred)
{
    if (global_canvas)
    {
        lv_canvas_set_px(global_canvas, x*100/2, CANVAS_HEIGHT-(y+1)*(CANVAS_HEIGHT/2), lv_color_hex(0xFF0000), LV_OPA_COVER);
        lv_canvas_set_px(global_canvas, x*100/2, CANVAS_HEIGHT-(y_pred+1)*(CANVAS_HEIGHT/2), lv_color_hex(0x00FF00), LV_OPA_COVER);
    }

    if (global_label2)
    {
        lv_label_set_text_fmt(global_label2, "x=%f\ny=%f\ny_pred=%f\ndelta=%f", x, y, y_pred, y_pred - y);
    }
}

static rt_device_t lcd_device;
void tflm_ui_task(void *args)
{
    rt_err_t ret = RT_EOK;
    rt_uint32_t ms;
    static rt_device_t touch_device;

    /* init littlevGL */
    ret = littlevgl2rtt_init(LCD_DEVICE_NAME);
    if (ret != RT_EOK)
    {
        return;
    }

    rt_kprintf("littlevGL init done!\n");

#if 0
    touch_device = rt_device_find(TOUCH_NAME);
    if(touch_device==RT_NULL)
    {
        rt_kprintf("touch_device!=NULL!");
        RT_ASSERT(0);
    }
    rt_device_control(touch_device, RTGRAPHIC_CTRL_POWEROFF, NULL);
#endif

    ret = tflm_ui_obj_init();
    if (ret != RT_EOK)
    {
        return;
    }
    rt_kprintf("tflm_ui_obj_init done!\n");

    while (1)
    {
        ms = lv_task_handler();
        rt_thread_mdelay(ms);
    }
}





