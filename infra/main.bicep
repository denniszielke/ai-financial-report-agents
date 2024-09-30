targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the the environment which is used to generate a short unique hash used in all resources.')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string
param aiResourceLocation string
@description('Id of the user or app to assign application roles')
param resourceGroupName string = ''
param containerAppsEnvironmentName string = ''
param containerRegistryName string = ''
param openaiName string = ''
param applicationInsightsDashboardName string = ''
param applicationInsightsName string = ''
param logAnalyticsName string = ''

var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))
var tags = { 'azd-env-name': environmentName, 'app': 'ai-agents', 'tracing': 'yes' }

param deployPostgres bool = false
param deploySessions bool = false
param completionDeploymentModelName string = 'gpt-4o'
param completionModelName string = 'gpt-4o'
param completionModelVersion string = '2024-08-06'
param embeddingDeploymentModelName string = 'text-embedding-3-small'
param embeddingModelName string = 'text-embedding-3-small'
param openaiApiVersion string = '2024-02-01'
param openaiCapacity int = 50
param modelDeployments array = [
  {
    name: completionDeploymentModelName
    model: {
      format: 'OpenAI'
      name: completionModelName
      version: completionModelVersion
    }
  }
  {
    name: embeddingDeploymentModelName
    model: {
      format: 'OpenAI'
      name: embeddingModelName
      version: '1'
    }
  }
]

var databaseAdmin = 'dbadmin'
var databaseName = 'langfuse'
@secure()
param databasePassword string = uniqueString(subscription().id, environmentName, location)

// Organize resources in a resource group
resource resourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: !empty(resourceGroupName) ? resourceGroupName : '${abbrs.resourcesResourceGroups}${environmentName}'
  location: location
  tags: tags
}

// Container apps host (including container registry)
module containerApps './core/host/container-apps.bicep' = {
  name: 'container-apps'
  scope: resourceGroup
  params: {
    name: 'app'
    containerAppsEnvironmentName: !empty(containerAppsEnvironmentName) ? containerAppsEnvironmentName : '${abbrs.appManagedEnvironments}${resourceToken}'
    containerRegistryName: !empty(containerRegistryName) ? containerRegistryName : '${abbrs.containerRegistryRegistries}${resourceToken}'
    location: location
    logAnalyticsWorkspaceName: monitoring.outputs.logAnalyticsWorkspaceName
    applicationInsightsName: monitoring.outputs.applicationInsightsName
    identityName: '${abbrs.managedIdentityUserAssignedIdentities}api-agents'
    openaiName: openai.outputs.openaiName
    dynamcSessionsName: dynamicSessions.outputs.name
  }
}

module dynamicSessions './core/host/dynamic-sessions.bicep' = if(deploySessions) {
  name: 'dynamic-${resourceToken}'
  scope: resourceGroup
  params: {
    name: 'sessions'
    location: location
    tags: tags
  }
}

// Azure OpenAI Model
module openai './ai/openai.bicep' = {
  name: 'openai'
  scope: resourceGroup
  params: {
    location: !empty(aiResourceLocation) ? aiResourceLocation : location
    tags: tags
    customDomainName: !empty(openaiName) ? openaiName : '${abbrs.cognitiveServicesAccounts}${resourceToken}'
    name: !empty(openaiName) ? openaiName : '${abbrs.cognitiveServicesAccounts}${resourceToken}'
    deployments: modelDeployments
    capacity: openaiCapacity
  }
}

module search './ai/search.bicep' = {
  name: 'search'
  scope: resourceGroup
  params: {
    location: location
    tags: tags
    name: !empty(openaiName) ? openaiName : '${abbrs.searchSearchServices}${resourceToken}'
  }
}

// Monitor application with Azure Monitor
module monitoring './core/monitor/monitoring.bicep' = {
  name: 'monitoring'
  scope: resourceGroup
  params: {
    location: location
    tags: tags
    logAnalyticsName: !empty(logAnalyticsName) ? logAnalyticsName : '${abbrs.operationalInsightsWorkspaces}${resourceToken}'
    applicationInsightsName: !empty(applicationInsightsName) ? applicationInsightsName : '${abbrs.insightsComponents}${resourceToken}'
    applicationInsightsDashboardName: !empty(applicationInsightsDashboardName) ? applicationInsightsDashboardName : '${abbrs.portalDashboards}${resourceToken}'
  }
}

module postgresServer 'core/data/flexibleserver.bicep' = if(deployPostgres) {
  name: 'postgresql'
  scope: resourceGroup
  params: {
    name: '${abbrs.dBforPostgreSQLServers}${resourceToken}'
    location: location
    tags: tags
    sku: {
      name: 'Standard_B1ms'
      tier: 'Burstable'
    }
    storage: {
      storageSizeGB: 32
    }
    version: '16'
    administratorLogin: databaseAdmin
    administratorLoginPassword: databasePassword
    databaseNames: [ databaseName ]
    allowAzureIPsFirewall: true
  }
}

// module langfuse 'app/fuse.bicep' = if(deployPostgres) {
//   name: 'langfuse'
//   scope: resourceGroup
//   params: {
//     name: 'langfuse'
//     location: location
//     tags: tags 
//     identityName: '${abbrs.managedIdentityUserAssignedIdentities}api-agents'
//     postgresServerFqdn: postgresServer.outputs.fqdn
//     databaseName: databaseName
//     databaseAdmin: databaseAdmin
//     databasePassword: databasePassword
//   }
// }

output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = tenant().tenantId
output AZURE_RESOURCE_GROUP string = resourceGroup.name

output APPLICATIONINSIGHTS_CONNECTION_STRING string = monitoring.outputs.applicationInsightsConnectionString
output APPLICATIONINSIGHTS_NAME string = monitoring.outputs.applicationInsightsName
output AZURE_CONTAINER_ENVIRONMENT_NAME string = containerApps.outputs.environmentName
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerApps.outputs.registryLoginServer
output AZURE_CONTAINER_REGISTRY_NAME string = containerApps.outputs.registryName
output OPENAI_API_TYPE string = 'azure'
output AZURE_OPENAI_VERSION string = openaiApiVersion
output OPENAI_API_VERSION string = openaiApiVersion
output AZURE_OPENAI_API_KEY string = openai.outputs.openaiKey
output AZURE_OPENAI_ENDPOINT string = openai.outputs.openaiEndpoint
output AZURE_OPENAI_COMPLETION_MODEL string = completionModelName
output AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME string = completionDeploymentModelName
output AZURE_OPENAI_EMBEDDING_MODEL string = embeddingModelName
output AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME string = embeddingDeploymentModelName
output POOL_MANAGEMENT_ENDPOINT string = dynamicSessions.outputs.poolManagementEndpoint
output AZURE_AI_SEARCH_NAME string = search.outputs.searchName
output AZURE_AI_SEARCH_ENDPOINT string = search.outputs.searchEndpoint
output AZURE_AI_SEARCH_KEY string = search.outputs.searchAdminKey