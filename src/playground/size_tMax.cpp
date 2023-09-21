#include <iostream>
#include <limits>

std::string formatSize(size_t size)
{
  double gbSize = static_cast<double>(size) / (1024 * 1024 * 1024);
  // 将大小格式化为带有两位小数的字符串
  std::string formattedSize = std::to_string(gbSize);
  size_t decimalPos = formattedSize.find('.');
  if (decimalPos != std::string::npos && decimalPos + 3 < formattedSize.size())
  {
    formattedSize = formattedSize.substr(0, decimalPos + 3);
  }

  // 添加单位
  formattedSize += " GB";

  return formattedSize;
}

int main()
{

  size_t maxValue = std::numeric_limits<size_t>::max();
  std::cout << "size_t max value: " << formatSize(maxValue) << std::endl;
  return 0;
}