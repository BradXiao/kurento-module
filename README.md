# AI (Object Detection) Module for KMS

This is AI (object detection) module for KMS.

![](architecture.png|width=500)

## Pre-built binaries

### Prerequites

- [Kurento Media Server](https://github.com/BradXiao/kurento-ubuntu) installed
- a serialized tensorRT model
  - [Yolov7](https://github.com/WongKinYiu/yolov7)

> [!NOTE]
> Since TensorRT models are not portable across devices, platforms or versions, it is not provided. You have to build it on your own.


### Module fast installation
The script will install the module and Java client library into local maven repository.

```bash
./install.sh
```


## Usage
### Specify model paths in the config file

The `model_pool_config.json` file is in `assets` in this repository. Specify your model path and other paramters according to your need.
```json
{
    "device_id": 0,
    "default_model_name": "yolov7",
    "models": [
        {
            "enabled": true,
            "name": "yolov7",
            "max_model_limit": 2,
            "model_abs_path": "/your/path/yolov7.trt"
        },
        {
...
```


### Add the config path to your Kurento Media Server service file

The default path is `/etc/systemd/system/kms.service`. Add a new environment line specifying `OBJDET_CONFIG=/your/path/model_pool_config.json`.

```ini
...
 Environment="KURENTO_LOG_FILE_SIZE=20"
 Environment="KURENTO_NUMBER_LOG_FILES=10"
 Environment="KURENTO_LOGS_PATH=/var/log/kurento-media-server/"
 Environment="OBJDET_CONFIG=/your/path/model_pool_config.json"
 ExecStart=/opt/kms/bin/kurento-media-server
 Restart=always
...
```

### Develop your own Java web app (optional)
Add the Java client library to your `pom.xml`.

```xml
...
<dependency>
    <groupId>org.kurento.module</groupId>
    <artifactId>objdet</artifactId>
    <version>1.0.0</version>
</dependency>
...
```

For more information, please refer to [Kurento-AP](https://github.com/BradXiao/kurento-ap).


## Disclaimer

THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

